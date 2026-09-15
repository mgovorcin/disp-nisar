"""NISAR GSLC → Zarr converter and ZarrStack reader.

Zarr store layout
-----------------
<output>.zarr/
  freqA/
    HH    : (n_dates, ny, nx_a)  complex64
    mask  : (n_dates, ny, nx_a)  uint8
    .attrs: dates, xCoordinates, yCoordinates, xCoordinateSpacing,
            yCoordinateSpacing, centerFrequency, projection
  freqB/              [optional]
    HH    : (n_dates, ny, nx_b)  complex64
    mask  : (n_dates, ny, nx_b)  uint8
  .zattrs : dates, file_list

Chunk shape is (n_dates, spatial_chunks, spatial_chunks) — all dates per
spatial tile, which is optimal for phase-linking (reads one chunk per block).

Usage
-----
    from disp_nisar._zarr import gslc_to_zarr, ZarrStack

    # one-time preprocessing (~16 GB/file × n_dates disk write)
    gslc_to_zarr(files, 'stack.zarr', freqs=('A',))

    # drop-in for VRTStack anywhere in the dolphin workflow
    stack = ZarrStack('stack.zarr', freq='A')
    block = stack[:, 0:512, 0:512]          # numpy array, shape (n_dates, 512, 512)
    da_arr = stack.as_dask()                # dask array for parallel ops
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(fname: str | Path) -> str:
    """Extract YYYYMMDD from a NISAR GSLC filename."""
    m = re.search(r"_(\d{8})T\d{6}_", Path(fname).name)
    if m:
        return m.group(1)
    # fallback: first 8-digit sequence
    m = re.search(r"(\d{8})", Path(fname).name)
    return m.group(1) if m else "unknown"


def _make_compressor(name: str = "lz4"):
    """Return a zarr v3 BloscCodec for the requested algorithm."""
    from zarr.codecs import BloscCodec  # zarr ≥ 3.x

    cname_map = {"lz4": "lz4", "zstd": "zstd", "lz4hc": "lz4hc", "none": None}
    cname = cname_map.get(name, "lz4")
    if cname is None:
        return []
    # clevel=1 is ~3× faster to encode than clevel=5 with near-identical ratio
    # on complex SAR data (lz4 plateaus early).
    return [BloscCodec(cname=cname, clevel=1, shuffle="shuffle")]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def gslc_to_zarr(
    file_list: Sequence[str | Path],
    output_path: str | Path,
    freqs: Sequence[str] = ("A",),
    pol: str = "HH",
    spatial_chunks: int = 2048,
    row_batch: int = 2048,
    num_workers: int = 0,
    compressor: str = "lz4",
    blosc_threads: int = 0,
    overwrite: bool = False,
    progress: bool = True,
) -> Any:
    """Convert NISAR GSLC HDF5 files to a single Zarr store.

    Parameters
    ----------
    file_list:
        Chronologically sorted list of NISAR GSLC .h5 files.
    output_path:
        Output Zarr store path, e.g. ``'stack.zarr'``.
    freqs:
        Frequencies to extract: ``('A',)`` or ``('A', 'B')``.
    pol:
        Polarization, default ``'HH'``.
    spatial_chunks:
        Spatial chunk size in pixels (both x and y). The time dimension is
        always chunked as ``n_dates`` so every phase-linking block is one chunk.
        Should match the phase-linking block size (default 2048).
    row_batch:
        Number of HDF5 rows to read at once.  Must equal ``spatial_chunks``
        to avoid partial-chunk read-merge-write cycles on write.
        Controls peak memory: ``n_dates × row_batch × nx × 8 bytes``
        (e.g. 4 dates × 2048 × 70 K cols ≈ 4.6 GB).
    num_workers:
        Number of threads to read HDF5 files in parallel.  ``0`` (default)
        means use ``n_dates`` threads — one per file.  h5py releases the GIL
        during I/O so threads genuinely overlap.  Set to ``1`` to disable.
    compressor:
        Blosc compressor: ``'lz4'`` (fast), ``'zstd'`` (best ratio), ``'none'``.
    blosc_threads:
        Number of internal threads Blosc uses to compress each chunk.
        ``0`` (default) picks ``min(8, os.cpu_count()//2)``. Parallel compression
        is the dominant cost when ``compressor != 'none'``.
    overwrite:
        Recreate the store if it already exists.
    progress:
        Show a tqdm progress bar.

    Returns
    -------
    zarr.Group
        Opened root group of the written store (read-only).

    Notes
    -----
    For 6 dates of freqA, the uncompressed stack is ~222 GB.  Blosc-lz4
    typically achieves ~2–3× on complex SAR data, so expect ~80–110 GB on disk.
    Full write time depends on I/O bandwidth; ~30–90 min on a fast NFS/SSD.

    """
    import os
    import queue
    from concurrent.futures import ThreadPoolExecutor
    from threading import Thread

    import h5py
    import zarr

    try:
        from tqdm.auto import tqdm
    except ImportError:

        def tqdm(it, **_kw):
            return it

    file_list = [Path(f) for f in file_list]
    n_dates = len(file_list)
    dates = [_parse_date(f) for f in file_list]
    output_path = Path(output_path)
    n_threads = n_dates if num_workers == 0 else max(1, num_workers)

    # Fix 5: parallelize Blosc compression. Try blosc2 first (zarr-v3 default),
    # then plain blosc. Silently no-op if neither exposes set_nthreads.
    if blosc_threads <= 0:
        blosc_threads = max(1, min(8, (os.cpu_count() or 2) // 2))
    for _mod_name in ("blosc2", "blosc"):
        try:
            _mod = __import__(_mod_name)
            _mod.set_nthreads(blosc_threads)
            print(f"{_mod_name}.set_nthreads({blosc_threads})")
            break
        except (ImportError, AttributeError):
            continue

    if output_path.exists():
        if not overwrite:
            print(f"{output_path} already exists — returning existing store.")
            print("Pass overwrite=True to recreate.")
            return zarr.open_group(str(output_path), mode="r")
        import shutil

        shutil.rmtree(output_path)

    codecs = _make_compressor(compressor)
    root = zarr.open_group(str(output_path), mode="w")
    root.attrs["dates"] = dates
    root.attrs["n_dates"] = n_dates
    root.attrs["file_list"] = [str(f) for f in file_list]

    # Keep all HDF5 files open for the duration of the write
    handles = [h5py.File(f, "r") for f in file_list]

    def _read_one(args: tuple) -> tuple:
        """Read one date's SLC + mask rows from its open HDF5 handle."""
        i, h, grp_path, pol, row_start, row_end = args
        slc = h[f"{grp_path}/{pol}"][row_start:row_end, :]
        msk = h[f"{grp_path}/mask"][row_start:row_end, :]
        return i, slc, msk

    # Fix 4: producer/consumer so read+mask of batch N+1 overlaps with
    # compress+write of batch N. maxsize=1 caps memory at ~3 batches in flight
    # (one being written, one in queue, one being read).
    write_q: queue.Queue = queue.Queue(maxsize=1)
    writer_error: list[BaseException] = []

    def _writer() -> None:
        while True:
            item = write_q.get()
            try:
                if item is None:
                    return
                rs, re_, sbuf, mbuf, sarr, marr = item
                sarr[:, rs:re_, :] = sbuf
                marr[:, rs:re_, :] = mbuf
            except BaseException as exc:  # noqa: BLE001
                writer_error.append(exc)
                return
            finally:
                write_q.task_done()

    writer_thread = Thread(target=_writer, name="zarr-writer", daemon=True)
    writer_thread.start()

    try:
        for freq_letter in freqs:
            freq_key = f"frequency{freq_letter.upper()}"
            grp_path = f"science/LSAR/GSLC/grids/{freq_key}"

            ds0 = handles[0][f"{grp_path}/{pol}"]
            ny, nx = ds0.shape
            dtype = ds0.dtype

            # Read coordinate metadata from first file
            meta: dict[str, Any] = {}
            for mkey in [
                "xCoordinates",
                "yCoordinates",
                "xCoordinateSpacing",
                "yCoordinateSpacing",
                "centerFrequency",
                "projection",
            ]:
                node = handles[0][grp_path].get(mkey)
                if node is None:
                    continue
                val = node[()]
                meta[mkey] = val.tolist() if hasattr(val, "tolist") else float(val)

            freq_grp = root.require_group(f"freq{freq_letter.upper()}")
            freq_grp.attrs.update(meta)
            freq_grp.attrs["dates"] = dates
            freq_grp.attrs["pol"] = pol
            freq_grp.attrs["shape_yx"] = [ny, nx]

            chunks = (n_dates, spatial_chunks, spatial_chunks)

            slc_arr = freq_grp.create_array(
                pol,
                shape=(n_dates, ny, nx),
                chunks=chunks,
                dtype=dtype,
                compressors=codecs,
            )
            mask_arr = freq_grp.create_array(
                "mask",
                shape=(n_dates, ny, nx),
                chunks=chunks,
                dtype="uint8",
                compressors=codecs,
            )

            raw_gb = n_dates * ny * nx * np.dtype(dtype).itemsize / 1e9
            print(
                f"freq{freq_letter.upper()}/{pol}: ({n_dates}, {ny}, {nx}) "
                f"{dtype}  raw={raw_gb:.1f} GB  chunks={chunks}  "
                f"threads={n_threads}"
            )

            row_starts = range(0, ny, row_batch)
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                for row_start in tqdm(
                    row_starts,
                    desc=f"freq{freq_letter.upper()}→zarr",
                    unit="batch",
                    disable=not progress,
                ):
                    row_end = min(row_start + row_batch, ny)
                    nrows = row_end - row_start

                    slc_buf = np.empty((n_dates, nrows, nx), dtype=dtype)
                    mask_buf = np.empty((n_dates, nrows, nx), dtype="uint8")

                    # Read all dates in parallel — each thread reads a
                    # different HDF5 file handle, so no locking needed.
                    args = [
                        (i, h, grp_path, pol, row_start, row_end)
                        for i, h in enumerate(handles)
                    ]
                    for i, slc, msk in pool.map(_read_one, args):
                        slc_buf[i] = slc
                        mask_buf[i] = msk

                    # NISAR GSLC mask: 1 = valid, 255 = nodata (never 0).
                    # Zero out nodata pixels so downstream JAX/cuSolver never
                    # receives NaN.  np.nan_to_num catches any residual NaNs in
                    # pixels that slipped through with valid-looking mask values.
                    slc_buf[mask_buf != 1] = 0
                    np.nan_to_num(slc_buf, copy=False)

                    # Hand off to background writer; blocks here if the writer
                    # is still busy with the previous batch (back-pressure).
                    if writer_error:
                        raise writer_error[0]
                    write_q.put(
                        (row_start, row_end, slc_buf, mask_buf, slc_arr, mask_arr)
                    )

            # Drain this frequency's queued writes before moving to the next
            # `freq_grp` so we don't interleave assignments to different arrays.
            write_q.join()
            if writer_error:
                raise writer_error[0]

    finally:
        # Stop writer cleanly, then close HDF5 handles.
        try:
            write_q.put(None)
            writer_thread.join(timeout=120)
        except Exception:
            pass
        for h in handles:
            try:
                h.close()
            except Exception:
                pass

    zarr.consolidate_metadata(str(output_path))
    print(f"\nDone → {output_path}")
    return zarr.open_group(str(output_path), mode="r")


# ---------------------------------------------------------------------------
# Reader — drop-in replacement for dolphin's VRTStack
# ---------------------------------------------------------------------------


class ZarrStack:
    """Read a Zarr SLC stack produced by :func:`gslc_to_zarr`.

    Implements the ``DatasetReader`` protocol expected by dolphin
    (``shape``, ``dtype``, ``ndim``, ``__getitem__``), so it can be passed
    anywhere a ``VRTStack`` is used.

    Parameters
    ----------
    zarr_path:
        Path to the ``.zarr`` store written by :func:`gslc_to_zarr`.
    freq:
        Frequency band: ``'A'`` or ``'B'``.
    pol:
        Polarization: ``'HH'`` or ``'HV'``.

    Examples
    --------
    >>> stack = ZarrStack('stack.zarr')
    >>> stack.shape                          # (n_dates, ny, nx)
    >>> block = stack[:, 256:768, 256:768]   # numpy array
    >>> da_arr = stack.as_dask()             # dask array, same shape
    >>> da_arr.rechunk({0: -1, 1: 256, 2: 256})  # rechunk for experiments

    """

    def __init__(
        self,
        zarr_path: str | Path,
        freq: str = "A",
        pol: str = "HH",
    ) -> None:
        import zarr

        self._path = Path(zarr_path)
        self._freq = freq.upper()
        self._pol = pol

        root = zarr.open_group(str(zarr_path), mode="r")
        freq_grp = root[f"freq{self._freq}"]
        self._arr = freq_grp[pol]  # zarr.Array (n_dates, ny, nx)
        self._meta = dict(freq_grp.attrs)
        self._root_meta = dict(root.attrs)

    # -- DatasetReader protocol ------------------------------------------------

    # Zarr reads are thread-safe; dolphin checks this attribute to skip the
    # global read_lock and allow all workers to read simultaneously.
    thread_safe = True

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._arr.dtype)

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self._arr[key])

    # -- VRTStack compatibility ------------------------------------------------

    @property
    def outfile(self) -> Path:
        """Mimic VRTStack.outfile — returns the zarr store path."""
        return self._path

    @property
    def file_list(self) -> list[Path]:
        return [Path(f) for f in self._root_meta.get("file_list", [])]

    @property
    def subdataset(self) -> str:
        """HDF5 subdataset path — for compatibility with dolphin's sequential.py."""
        return f"/science/LSAR/GSLC/grids/frequency{self._freq}/{self._pol}"

    @property
    def gdal_path(self) -> str:
        """GDAL-readable path to the last source HDF5 file.

        Use this wherever dolphin expects a ``like_filename`` that GDAL can
        open to read geotransform / projection / size metadata::

            dolphin.ps.create_ps(..., like_filename=zarr_stack.gdal_path)
        """
        src = self.file_list
        if not src:
            raise RuntimeError(
                "No source files in zarr metadata. "
                "Pass like_filename=<path_to_any_gslc_h5> explicitly."
            )
        return f'NETCDF:"{src[-1]}":{self.subdataset}'

    def __fspath__(self) -> str:
        """Make fspath(zarr_stack) return the GDAL-readable HDF5 path.

        This is what dolphin's io._core._get_gdal_ds calls via
        ``gdal.Open(fspath(like_filename))``.
        """
        return self.gdal_path

    # -- Extra helpers ---------------------------------------------------------

    @property
    def dates(self) -> list[str]:
        return self._meta.get("dates") or self._root_meta.get("dates") or []

    @property
    def n_dates(self) -> int:
        return self.shape[0]

    @property
    def mask(self) -> "ZarrMaskView":
        """Access the mask sub-array (same shape as SLC stack)."""
        return ZarrMaskView(self._path, self._freq)

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_coords, y_coords) 1-D arrays."""
        x = np.asarray(self._meta["xCoordinates"])
        y = np.asarray(self._meta["yCoordinates"])
        return x, y

    def as_dask(self, chunks: tuple | None = None) -> Any:
        """Return a dask array backed by this Zarr store.

        Parameters
        ----------
        chunks:
            Override chunk shape. Default uses the stored chunk shape
            ``(n_dates, spatial_chunks, spatial_chunks)``.

        """
        import dask.array as da

        da_arr = da.from_zarr(
            str(self._path), component=f"freq{self._freq}/{self._pol}"
        )
        if chunks is not None:
            da_arr = da_arr.rechunk(chunks)
        return da_arr

    def slice_dates(self, file_list: "Sequence[str | Path]") -> "ZarrStackSlice | None":
        """Return a time-sliced view for the given subset of source files.

        Used by ``run_wrapped_phase_sequential`` to create per-ministack
        readers that pull directly from the zarr store instead of re-reading
        the source HDF5 files.

        Returns ``None`` if any file in *file_list* is not present in this
        store (e.g. compressed SLCs produced by a previous ministack), in
        which case the caller should fall back to a VRTStack.
        """
        file_map = {str(f): i for i, f in enumerate(self.file_list)}
        indices: list[int] = []
        matched: list[Path] = []
        for f in file_list:
            idx = file_map.get(str(f))
            if idx is None:
                return None
            indices.append(idx)
            matched.append(Path(f))
        return ZarrStackSlice(self, indices, matched)

    def __repr__(self) -> str:
        ny, nx = self.shape[1], self.shape[2]
        return (
            f"ZarrStack(freq{self._freq}/{self._pol}  "
            f"shape=({self.n_dates}, {ny}, {nx})  "
            f"dtype={self.dtype}  dates={self.dates[0]}…{self.dates[-1]})"
        )


class ZarrStackSlice:
    """Time-sliced view of a :class:`ZarrStack` for per-ministack processing.

    Implements the same ``DatasetReader`` / ``VRTStack``-compatible protocol
    as ``ZarrStack`` but exposes only a subset of the time dimension.
    Passed as *vrt_stack* to ``run_wrapped_phase_single`` so that phase
    linking reads directly from zarr instead of re-opening the HDF5 files.
    """

    thread_safe = True

    def __init__(
        self,
        parent: "ZarrStack",
        indices: list[int],
        file_list: list[Path],
    ) -> None:
        self._parent = parent
        self._indices = indices
        self._file_list = file_list

    # -- DatasetReader protocol ------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._indices),) + self._parent.shape[1:]

    @property
    def dtype(self) -> np.dtype:
        return self._parent.dtype

    @property
    def ndim(self) -> int:
        return 3

    def __getitem__(self, key: Any) -> np.ndarray:
        # EagerLoader calls reader[:, row_slice, col_slice].
        # Map the time axis through self._indices.
        if isinstance(key, tuple) and len(key) == 3:
            t_key, row_key, col_key = key
        elif isinstance(key, tuple) and len(key) == 2:
            t_key, row_key = key
            col_key = slice(None)
        else:
            t_key = key
            row_key = col_key = slice(None)

        if t_key is Ellipsis or t_key == slice(None):
            idx: Any = self._indices
        elif isinstance(t_key, int):
            idx = self._indices[t_key]
        elif isinstance(t_key, slice):
            idx = self._indices[t_key]
        else:
            idx = [self._indices[i] for i in t_key]

        # oindex supports orthogonal (list-of-ints × slice × slice) reads
        return np.asarray(self._parent._arr.oindex[idx, row_key, col_key])

    # -- VRTStack compatibility ------------------------------------------------

    @property
    def outfile(self) -> str:
        """GDAL-readable path used as *like_filename* for output file setup."""
        return self._parent.gdal_path

    @property
    def file_list(self) -> list[Path]:
        return self._file_list

    @property
    def subdataset(self) -> str:
        return self._parent.subdataset

    def __fspath__(self) -> str:
        return self._parent.gdal_path

    def __repr__(self) -> str:
        return f"ZarrStackSlice({len(self._indices)} dates of {self._parent!r})"


class ZarrMaskView:
    """Thin wrapper for the mask array inside a ZarrStack store."""

    def __init__(self, zarr_path: Path, freq: str) -> None:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r")
        self._arr = root[f"freq{freq}"]["mask"]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self._arr[key])

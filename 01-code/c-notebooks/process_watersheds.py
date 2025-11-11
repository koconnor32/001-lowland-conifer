#!/usr/bin/env python3
"""
HUC8 Watershed DEM Extraction Script - Parallelized Version
"""

import os
import glob
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
import numpy as np
from typing import List, Tuple, Dict
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HUC8DEMProcessor:
    
    def __init__(self, countydems_dir: str, huc8_shapefile: str, output_dir: str):
        self.countydems_dir = Path(countydems_dir)
        self.huc8_shapefile = Path(huc8_shapefile)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading HUC8 watersheds from {self.huc8_shapefile}")
        self.watersheds = gpd.read_file(self.huc8_shapefile)
        logger.info(f"Loaded {len(self.watersheds)} watersheds")
        
        self.dem_files = self._get_dem_files()
        logger.info(f"Found {len(self.dem_files)} DEM files")
    
    def _get_dem_files(self) -> List[Path]:
        tif_files = list(self.countydems_dir.glob("**/*.tif"))
        tif_files = [f for f in tif_files if not f.name.endswith('.ovr')]
        return tif_files
    
    def _get_dem_coverage(self) -> Dict[Path, box]:
        logger.info("Computing DEM coverage areas...")
        dem_coverage = {}
        watershed_crs = self.watersheds.crs
        
        for dem_file in self.dem_files:
            try:
                with rasterio.open(dem_file) as src:
                    bounds = src.bounds
                    raster_crs = src.crs
                    
                    bounds_gdf = gpd.GeoDataFrame(
                        {'geometry': [box(*bounds)]},
                        crs=raster_crs
                    )
                    
                    if raster_crs != watershed_crs:
                        bounds_gdf = bounds_gdf.to_crs(watershed_crs)
                    
                    dem_coverage[dem_file] = bounds_gdf.geometry.iloc[0]
                    logger.info(f"  Processed coverage for {dem_file.name}")
            except Exception as e:
                logger.warning(f"Could not get coverage for {dem_file}: {e}")
        
        return dem_coverage
    
    def _filter_watersheds_by_coverage(self, dem_coverage: Dict[Path, box]) -> gpd.GeoDataFrame:
        logger.info("Filtering watersheds by DEM coverage...")
        
        from shapely.ops import unary_union
        all_dem_coverage = unary_union(list(dem_coverage.values()))
        
        filtered = self.watersheds[self.watersheds.geometry.intersects(all_dem_coverage)]
        
        logger.info(f"Filtered to {len(filtered)} watersheds (from {len(self.watersheds)} total)")
        
        return filtered
    
    def process_all_watersheds(self, n_processes: int = 8, resume: bool = True):
        logger.info(f"Starting parallel processing with {n_processes} processes")
        
        dem_coverage = self._get_dem_coverage()
        
        if not dem_coverage:
            logger.error("No valid DEM files found!")
            return
        
        filtered_watersheds = self._filter_watersheds_by_coverage(dem_coverage)
        
        if len(filtered_watersheds) == 0:
            logger.warning("No watersheds overlap with available DEMs!")
            return
        
        watershed_data = []
        skipped = 0
        
        for idx, row in filtered_watersheds.iterrows():
            huc_id = None
            for field in ['HUC8', 'huc8', 'HUC_8', 'HUC_CODE', 'HUC', 'NAME']:
                if field in row:
                    huc_id = str(row[field])
                    break
            
            if huc_id is None:
                huc_id = f"watershed_{idx}"
            
            output_file = self.output_dir / f"{huc_id}_dem.tif"
            if resume and output_file.exists():
                skipped += 1
                continue
            
            overlapping_dems = []
            for dem_path, dem_geom in dem_coverage.items():
                if row.geometry.intersects(dem_geom):
                    overlapping_dems.append(dem_path)
            
            if not overlapping_dems:
                continue
            
            watershed_data.append({
                'idx': idx,
                'huc_id': huc_id,
                'geometry': row.geometry,
                'crs': filtered_watersheds.crs,
                'overlapping_dems': overlapping_dems
            })
        
        if skipped > 0:
            logger.info(f"Skipping {skipped} watersheds that already have output files")
        
        if not watershed_data:
            logger.info("All watersheds already processed!")
            return
        
        logger.info(f"Processing {len(watershed_data)} watersheds")
        
        process_func = partial(
            process_single_watershed_fast,
            output_dir=self.output_dir,
            total=len(filtered_watersheds)
        )
        
        try:
            with Pool(processes=n_processes) as pool:
                results = pool.map(process_func, watershed_data)
            
            successful = sum(1 for r in results if r)
            logger.info(f"Processing complete! {successful}/{len(results)} watersheds processed successfully")
        
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user. Progress has been saved.")
            logger.info("Run the script again to resume from where you left off.")
            raise


def process_single_watershed_fast(watershed_info: dict, output_dir: Path, total: int) -> bool:
    idx = watershed_info['idx']
    huc_id = watershed_info['huc_id']
    watershed_geom = watershed_info['geometry']
    watershed_crs = watershed_info['crs']
    overlapping_dems = watershed_info['overlapping_dems']
    
    logger.info(f"Processing watershed {idx+1}/{total}: {huc_id}")
    logger.info(f"  [{huc_id}] Using {len(overlapping_dems)} overlapping DEMs")
    
    try:
        output_file = output_dir / f"{huc_id}_dem.tif"
        
        clip_and_merge_dems(
            overlapping_dems,
            watershed_geom,
            watershed_crs,
            output_file
        )
        logger.info(f"  [{huc_id}] Successfully created {output_file.name}")
        return True
        
    except Exception as e:
        logger.error(f"  [{huc_id}] Error processing watershed: {e}")
        return False


def clip_and_merge_dems(dem_files: List[Path], watershed_geom, 
                       watershed_crs, output_path: Path):
    if not dem_files:
        logger.warning(f"No DEMs found for {output_path.name}")
        return
    
    clipped_datasets = []
    temp_files = []
    
    reference_crs = None
    
    try:
        for idx, dem_file in enumerate(dem_files):
            with rasterio.open(dem_file) as src:
                if reference_crs is None:
                    reference_crs = src.crs
                
                watershed_in_dem_crs = gpd.GeoDataFrame(
                    {'geometry': [watershed_geom]},
                    crs=watershed_crs
                ).to_crs(src.crs)
                
                out_image, out_transform = mask(
                    src,
                    watershed_in_dem_crs.geometry,
                    crop=True,
                    nodata=src.nodata if src.nodata is not None else -9999
                )
                
                needs_reproject = (src.crs != reference_crs)
                
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "lzw",
                    "BIGTIFF": "YES",
                    "TILED": "YES"
                })
                
                if len(dem_files) > 1:
                    temp_path = output_path.parent / f"temp_clip_{os.getpid()}_{dem_file.stem}.tif"
                    temp_files.append(temp_path)
                    
                    with rasterio.open(temp_path, 'w', **out_meta) as dest:
                        dest.write(out_image)
                    
                    if needs_reproject:
                        logger.info(f"  Reprojecting {dem_file.name} to match reference CRS")
                        temp_reproj_path = output_path.parent / f"temp_reproj_{os.getpid()}_{dem_file.stem}.tif"
                        temp_files.append(temp_reproj_path)
                        
                        with rasterio.open(temp_path) as src_temp:
                            transform, width, height = calculate_default_transform(
                                src_temp.crs, reference_crs, 
                                src_temp.width, src_temp.height, 
                                *src_temp.bounds
                            )
                            
                            kwargs = src_temp.meta.copy()
                            kwargs.update({
                                'crs': reference_crs,
                                'transform': transform,
                                'width': width,
                                'height': height
                            })
                            
                            with rasterio.open(temp_reproj_path, 'w', **kwargs) as dst_temp:
                                for i in range(1, src_temp.count + 1):
                                    reproject(
                                        source=rasterio.band(src_temp, i),
                                        destination=rasterio.band(dst_temp, i),
                                        src_transform=src_temp.transform,
                                        src_crs=src_temp.crs,
                                        dst_transform=transform,
                                        dst_crs=reference_crs,
                                        resampling=Resampling.bilinear
                                    )
                        
                        clipped_datasets.append(rasterio.open(temp_reproj_path))
                    else:
                        clipped_datasets.append(rasterio.open(temp_path))
                else:
                    with rasterio.open(output_path, 'w', **out_meta) as dest:
                        dest.write(out_image)
                    return
        
        if len(clipped_datasets) > 1:
            logger.info(f"Merging {len(clipped_datasets)} DEMs")
            mosaic, out_trans = merge(clipped_datasets)
            
            out_meta = clipped_datasets[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw",
                "BIGTIFF": "YES",
                "TILED": "YES"
            })
            
            with rasterio.open(output_path, 'w', **out_meta) as dest:
                dest.write(mosaic)
    
    finally:
        for ds in clipped_datasets:
            ds.close()
        
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass


def main():
    COUNTYDEMS_DIR = "test_demclip"
    HUC8_SHAPEFILE = "huc8_buffer_5km.gpkg"
    OUTPUT_DIR = "watershed_dems_output"
    
    N_PROCESSES = 8
    
    processor = HUC8DEMProcessor(COUNTYDEMS_DIR, HUC8_SHAPEFILE, OUTPUT_DIR)
    processor.process_all_watersheds(n_processes=N_PROCESSES, resume=True)


if __name__ == "__main__":
    main()

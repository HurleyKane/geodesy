import sys
print(sys.path)
from geodesy.io.reader_tiff import read_tiff, read_tiffs
try:
    from geodesy.io.read_cn_faults import (read_gmt, CN_fault, CN_border_L1, CN_border_La,
                                           CN_block_L1, CN_block_L2, CN_block_L1_deduced,
                                           geo3al
                                           )
except:
    pass

__all__ = [
    'read_tiff',
    'read_tiffs',
    'read_gmt',
    'CN_fault',
    'CN_border_L1',
    'CN_border_La',
    'CN_block_L1',
    'CN_block_L2',
    'CN_block_L1_deduced',
    'geo3al'
]
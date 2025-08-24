
from collections import namedtuple

# Define genotype structure
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# Updated list of primitive operations
PRIMITIVES = [
    'none',              # No operation
    'avg_pool_3x3',      # Average pooling with 3x3 kernel
    'max_pool_3x3',      # Max pooling with 3x3 kernel
    'max_pool_3x1',      # Max pooling with 3x1 kernel
    'max_pool_5x1',      # Max pooling with 5x1 kernel
    'skip_connect',      # Skip connection (identity map)
    'sep_conv_3x1',      # Separable convolution with 3x1 kernel
    'sep_conv_5x1',      # Separable convolution with 5x1 kernel
    'sep_conv_1x3',      # Separable convolution with 1x3 kernel
    'sep_conv_1x5',      # Separable convolution with 1x5 kernel
    'dil_conv_3x1',      # Dilation convolution with 3x1 kernel
    'dil_conv_5x1',      # Dilation convolution with 5x1 kernel
    'conv_1x1',          # Regular convolution with 1x1 kernel
    'conv_3x3',          # Regular convolution with 3x3 kernel
]

# Default empty placeholder genotype
DARTS_GENOTYPE = Genotype(
    normal=[],
    normal_concat=[],
    reduce=[],
    reduce_concat=[]
)


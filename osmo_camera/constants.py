RAW_BIT_DEPTH = 2**10  # used for normalizing DN to DNR
COLOR_CHANNEL_INDICES = [0, 1, 2]

# https://docs.google.com/document/d/1yS73HuP4B1_bWbD2lI5cMBV5K6DHEkGePcHr8gBs-EE/edit#
# TL;DR We save signed data (requires 1 bit) and 6 bits of “padding” around the upper end of the range.
# 32 - 1 - 6 = 25.
DNR_TO_TIFF_FACTOR = 2**25

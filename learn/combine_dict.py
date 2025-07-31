# =============================================================================
# NeuSpeech Institute, NeuGaze Project
# Copyright (c) 2024 Yiqian Yang
#
# This code is part of the NeuGaze project developed at NeuSpeech Institute.
# Author: Yiqian Yang
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit:
# http://creativecommons.org/licenses/by-nc/4.0/
# =============================================================================

def combine_dicts(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = v
    return dict1


d1={'h':True}
d2={'v':True}
d1=combine_dicts(d1,d2)
print(d1)


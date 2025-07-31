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



class A:
    def __init__(self, args):
        self.args = args
        self.cls=None
        self.v='a'

    def give_observer(self,cls):
        self.cls = cls


class B:
    def __init__(self, cls):
        self.cls = cls
        self.v='b'

A=A('a')
B=B(A)
A.give_observer(B)
print(A.cls.v,B.cls.v)
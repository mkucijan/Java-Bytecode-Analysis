
import numpy as np

class Additional_Parameters():
    def __init__(self, args_dim = None, bidirectional=True, nonlinear=True, encode_int = True):
        self.args_dim = args_dim
        self.bidirectional = bidirectional
        self.nonlinear = nonlinear
        self.encode_int = encode_int


class Accuracy(object):
    
    def __init__(self):
        self.TP_TN=0.
        self.num_units=0.
        self.saved_acc=[]
        self.saved_steps=[]
    
    def reset(self):
        self.TP_TN=0.
        self.num_units=0.
    
    def evaluate(self, logits=None, labels=None, seq_len=None):
        if not logits is None:
            for lo, la, sl in zip(logits, labels, seq_len):
                sl = int(sl)
                pr = lo[:sl]
                fl = la[:sl]
                self.num_units += sl
                pr_ar = np.argmax(pr, axis=1).astype(np.int32)
                fl_ar = np.argmax(fl, axis=1).astype(np.int32)
                self.TP_TN += np.sum(pr_ar==fl_ar)
        
        if self.num_units:
            return self.TP_TN/self.num_units
    
    def save(self, step, reset=True):
        self.saved_acc.append(self.evaluate())
        self.saved_steps.append(step)
        if reset:
            self.reset()
    
        
    def __call__(self, **kwargs):
        return self.evaluate(kwargs)
    
    def __str__(self):
        if self.num_units:
            return str(self.TP_TN/self.num_units)
        else:
            return str(0.0)
from time import process_time
from datetime import timedelta

class StopWatch:
    def __init__(self):
        self.start()
    
    def __call__(self):
        self.lap()
    
    def start(self):
        self.tic = process_time()
    
    def stop(self, display=True):
        self.toc = process_time()
        self.elapsed_time = timedelta(seconds = self.toc - self.tic)
        if display:
            self.show()
    
    def lap(self):
        lap_time = timedelta(seconds = process_time() - self.tic)
        self.show(elapsed_time = lap_time, text = "lap time")
    
    def show(self, elapsed_time=None, text='elapsed time'):
        if type(elapsed_time) == type(None):
            elapsed_time = self.elapsed_time
        print('{0} = {1}\n'.format(text, elapsed_time))
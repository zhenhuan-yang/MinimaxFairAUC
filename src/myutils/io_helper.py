import os
import sys
import psutil

PROCESS = psutil.Process(os.getpid())

def mem_usage():
    return PROCESS.memory_info().rss//10**6

def make_file_path(root, *args, **kwargs):
    '''
    :param root: root folder
    :param args:
    : param kwargs:
    :return:
    '''
    if not os.path.exists(root):
        os.makedirs(root)
    file_name = ""
    if args:
        i = 0
        while i < len(args) - 1:
            file_name += args[i] + "_"
            i += 1
        file_name += args[-1]
    if kwargs:
        for arg, val in kwargs.items():
            file_name += arg + "_" + str(val) + "_"
        file_name = file_name[:-1]
    file_path = os.path.join(root, file_name)
    return file_path

class Logger(object):
    # https://stackoverflow.com/questions/10019456/usage-of-sys-stdout-flush-method
    def __init__(self, log_path=""):
        self.log_path = log_path
        if self.log_path:
            open(self.log_path, 'w').close()

        # while os.path.isfile(self.log_path):
        #     self.log_path += '+'
        # # print("Log file path:\n" + self.log_path)

    def log(self, string, newline=True, verbose=True):
        
        if self.log_path:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

        if verbose:
            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()

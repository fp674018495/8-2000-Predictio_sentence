import logging
import logging.handlers
import os 

class Logger():
    def __init__(self,filename,level=logging.INFO):
        file_dir=os.path.split(filename)[0]
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        format_str = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        self.logger=logging.getLogger()
        self.logger.setLevel(level)
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = logging.handlers.TimedRotatingFileHandler(filename=filename,
                                                       when='D',
                                                       interval=1,
                                                       backupCount=5,
                                                       encoding='utf-8'
                                                       )

        th.suffix = "%Y-%m-%d.log"
        th.setFormatter(format_str)
        self.logger.addHandler(th)
        self.logger.addHandler(sh)
    def remove_handler(self):
        dd= self.logger.handlers.copy()
        for i in dd:
            self.logger.removeHandler(i)

    def rebuild_log(self,filename):
        self.remove_handler()
        format_str = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        fp =logging.FileHandler(filename,encoding="utf-8")
        fp.setFormatter(format_str)
        self.logger.addHandler(fp)

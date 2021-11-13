import datetime

class TimeLogger():
    def __init__(self):
        # self._start_time = None
        # self._finish_time = None
        self._time_table = dict()

        return

    # log the time when a code start to run
    def set_start_time(self, time_id):
        #self._start_time = datetime.datetime.now()
        self._time_table[time_id] = dict()
        self._time_table[time_id]['start'] = datetime.datetime.now()
        print("time_id:", time_id, ", start time is logged")
        return

    # log the time when a code is finished
    def set_finish_time(self, time_id):
        #self._finish_time = datetime.datetime.now()
        self._time_table[time_id]['finish'] = datetime.datetime.now()
        print("time_id:", time_id, ", finish time is logged")
        return

    def get_delta(self, time_id, mode="raw"):
        # result_delta = self._finish_time - self._start_time
        result_delta = self._time_table[time_id]['finish'] - self._time_table[time_id]['start']
        if mode == "raw":
            return str(result_delta)
        elif mode == "minute" or mode == "min":
            return str(result_delta.seconds / 60)
        elif mode == "second" or mode == "sec":
            return str(result_delta.seconds)
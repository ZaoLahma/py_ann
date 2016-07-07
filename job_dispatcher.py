#!/usr/bin/env python
from multiprocessing import cpu_count
from threading import Lock
from threading import Thread 
from threading import Condition
from time import sleep

class EventJob():
    def __init__(self, event_listener, event_no, event_data):
        self.event_listener = event_listener
        self.event_no = event_no
        self.event_data = event_data
        
    def execute(self):
        self.event_listener.handle_event(self.event_no, self.event_data)

class Timer(Thread):
    def __init__(self, ms_to_sleep):
        Thread.__init__(self)
        self.ms_to_sleep = ms_to_sleep
        
    def run(self):
        sleep(self.ms_to_sleep / 1000)
        self.timer_func()

class EventTimer(Timer):
    def __init__(self, job_dispatcher, event_no, event_data, ms_to_sleep):
        Timer.__init__(self, ms_to_sleep)
        self.job_dispatcher = job_dispatcher
        self.event_no = event_no
        self.event_data = event_data
        
    def timer_func(self):
        self.job_dispatcher.raise_event(self.event_no, self.event_data)

class JobTimer(Timer):
    def __init__(self, job_dispatcher, job, ms_to_sleep):
        Timer.__init__(self, ms_to_sleep)
        self.job_dispatcher = job_dispatcher
        self.job = job
          
    def timer_func(self):
        self.job_dispatcher.execute_job(self.job)


class Worker(Thread):
    def __init__(self, JobDispatcher):
        Thread.__init__(self)
        self.is_busy = False
        self.jobQueue = JobDispatcher
        self.running = False
        self.cv = Condition(None)
        self.no_of_jobs_executed = 0
        self.busy_wait_cond = Condition(None)
        
    def run(self):
        while(self.running):
            self.is_busy = True
            job = self.jobQueue.get_job()
            if(None != job):
                self.no_of_jobs_executed += 1
                job.execute()
            else:
                if self.running:
                    with self.cv:
                        self.is_busy = False
                        with self.busy_wait_cond:
                            self.busy_wait_cond.notify()
                        self.cv.wait()
      
    def notify(self):
        with self.cv:
            self.cv.notify()          
            
    def start(self):
        self.running = True
        Thread.start(self)
        
    def stop(self):
        self.running = False
        self.notify()
        self.join()

    def get_is_busy(self):
        return self.is_busy
        
    def wait_until_not_busy(self):
        if self.is_busy:
            with self.busy_wait_cond:
                self.busy_wait_cond.wait()


class JobDispatcher(object):
    INSTANCE = None
    instance_creation_lock = Lock()    
    def __init__(self):
        if self.INSTANCE != None:
            raise ValueError("Attempted to create a second instance of the JobDispatcher")
        self.queue_index = 0
        self.queue_lock = Lock()
        self.subscribers_lock = Lock()
        self.active_queue = []
        self.free_queue = []
        self.no_of_cores = cpu_count()
        self.workers = []
        self.events_map = {}
     
    @classmethod
    def get_api(singelton_class):
        if None == singelton_class.INSTANCE:
            singelton_class.instance_creation_lock.acquire()
            if None == singelton_class.INSTANCE:
                singelton_class.INSTANCE = JobDispatcher()
            singelton_class.instance_creation_lock.release()
            
        return singelton_class.INSTANCE
     
    def stop(self, gracefully=False):
        index = 0
        
        for worker in self.workers:
            if True == gracefully:
                worker.wait_until_not_busy()
            worker.stop()
            print("Worker " + str(index) + " excuted " + str(worker.no_of_jobs_executed) + " jobs")        
            index += 1
            
    def get_job(self):
        self.queue_lock.acquire()
        if len(self.active_queue) == self.queue_index:
            self.active_queue = self.free_queue
            self.free_queue = []
            self.queue_index = 0
        
        if len(self.active_queue) == 0:
            self.queue_lock.release()
            return None
        else:
            curr_index = self.queue_index
            self.queue_index = self.queue_index + 1
            self.queue_lock.release()
            return self.active_queue[curr_index]
  
    def execute_job(self, job):
        self.free_queue.append(job)
        for worker in self.workers:
            if False == worker.get_is_busy():
                worker.notify()
                return
            
        #No free worker was found. Let's create a new one
        worker = Worker(self)
        worker.start()
        self.workers.append(worker)

    def execute_job_in(self, job, ms):
        job_timer = JobTimer(self, job, ms)
        job_timer.start()
            
    def subscribe_to_event(self, event_no, event_subscriber):
        self.subscribers_lock.acquire()
        if not event_no in self.events_map:
            self.events_map[event_no] = []
        self.events_map[event_no].append(event_subscriber)
        self.subscribers_lock.release()
        
    def unsubscribe_to_event(self, event_no, event_subscriber_to_remove):
        self.subscribers_lock.acquire()
        event_subscribers = None
        if event_no in self.events_map:
            event_subscribers = self.events_map[event_no]
            
        if None != event_subscribers:
            for event_subscriber in event_subscribers:
                if event_subscriber == event_subscriber_to_remove:
                    event_subscribers.remove(event_subscriber)
        self.subscribers_lock.release()
        
    def raise_event(self, event_no, event_data):
        self.subscribers_lock.acquire()
        event_subscribers = None
        if event_no in self.events_map:
            event_subscribers = self.events_map[event_no]
            
        if None != event_subscribers:
            for event_subscriber in event_subscribers:
                self.execute_job(EventJob(event_subscriber, event_no, event_data))
        self.subscribers_lock.release()
                
    def raise_event_in(self, event_no, event_data, ms):
        event_timer = EventTimer(self, event_no, event_data, ms)
        event_timer.start()

import redis
import pickle
from time import sleep
from os import getpid

class Queue(object):
	"""docstring for Queue"""
	def __init__(self, maxsize, name='default_name'):
		self.maxsize = maxsize
		self.name = name
		self.conn = redis.StrictRedis(host='localhost', port=6379, db=0)
		self.conn.flushall()

	def get(self, block=True, timeout=0):
		data = self.conn.blpop(self.name)
		return pickle.loads(data[1])

	def put(self, data, block=True, timeout=0):
		# Wait for get operations to decrease the queue size
		# There appears to be a race condition here. Two processes might read the queue size as 99, and then both push.
		# Ideally there should be some sort of mutex here to prevent that from happening
		while self.qsize() >= self.maxsize:
			print(getpid(), "Waiting for queue to free up")
			sleep(1)
		self.conn.rpush(self.name, pickle.dumps(data))

	def qsize(self):
		return self.conn.llen(self.name)

	def empty(self):
		return (self.qsize() == 0)

	def full(self):
		return (self.qsize() == self.maxsize)

from FIFORedisQueue import Queue
from multiprocessing import Process
from time import sleep

def enqueue(q):
	for x in range(5):
		if q.full():
			break
		val = {'f': x}
		q.put(val)
		sleep(.1)
	return

q = Queue(50, name='mine')
print(q.qsize())

p = Process(target=enqueue, args=[q])
p.start()

while q.empty() == False or p.is_alive():
	sleep(1)
	print(q.get()['f'], q.empty(), p.is_alive(), q.qsize())

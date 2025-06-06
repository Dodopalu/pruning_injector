import time



t0 = time.localtime()
time.sleep(1)  # Simulate some processing time
t1 = time.localtime()


timestamp0 = time.mktime(t0)
timestamp1 = time.mktime(t1)


diff_seconds = timestamp1 - timestamp0

print(diff_seconds)


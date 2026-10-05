from contextlib import contextmanager
import fcntl,time
@contextmanager
def slot(root,prefix,count,waiting=lambda:None):
 root.mkdir(parents=True,exist_ok=True);held=None;last=0
 while held is None:
  for i in range(count):
   f=(root/f'{prefix}{i}.lock').open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held=f;break
   except BlockingIOError:f.close()
  if held is None:
   if time.monotonic()-last>30:waiting();last=time.monotonic()
   time.sleep(2)
 try:yield
 finally:fcntl.flock(held,fcntl.LOCK_UN);held.close()

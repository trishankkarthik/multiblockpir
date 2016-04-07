# Shameless copy from Python 3.
# http://hg.python.org/cpython/file/default/Lib/threading.py

# TODO: Only take effect if we are not on Python 3.


from threading import Lock, RLock, _allocate_lock, _deque
from itertools import islice as _islice


class Condition:

    def __init__(self, lock=None):
        if lock is None:
            lock = RLock()
        self._lock = lock
        # Export the lock's acquire() and release() methods
        self.acquire = lock.acquire
        self.release = lock.release
        # If the lock defines _release_save() and/or _acquire_restore(),
        # these override the default implementations (which just call
        # release() and acquire() on the lock).  Ditto for _is_owned().
        try:
            self._release_save = lock._release_save
        except AttributeError:
            pass
        try:
            self._acquire_restore = lock._acquire_restore
        except AttributeError:
            pass
        try:
            self._is_owned = lock._is_owned
        except AttributeError:
            pass
        self._waiters = _deque()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __repr__(self):
        return "<Condition(%s, %d)>" % (self._lock, len(self._waiters))

    def _release_save(self):
        self._lock.release()           # No state to save

    def _acquire_restore(self, x):
        self._lock.acquire()           # Ignore saved state

    def _is_owned(self):
        # Return True if lock is owned by current_thread.
        # This method is called only if __lock doesn't have _is_owned().
        if self._lock.acquire(0):
            self._lock.release()
            return False
        else:
            return True

    def wait(self, timeout=None):
        if not self._is_owned():
            raise RuntimeError("cannot wait on un-acquired lock")
        waiter = _allocate_lock()
        waiter.acquire()
        self._waiters.append(waiter)
        saved_state = self._release_save()
        try:    # restore state no matter what (e.g., KeyboardInterrupt)
            if timeout is None:
                waiter.acquire()
                gotit = True
            else:
                if timeout > 0:
                    gotit = waiter.acquire(True, timeout)
                else:
                    gotit = waiter.acquire(False)
                if not gotit:
                    try:
                        self._waiters.remove(waiter)
                    except ValueError:
                        pass
            return gotit
        finally:
            self._acquire_restore(saved_state)

    def wait_for(self, predicate, timeout=None):
        endtime = None
        waittime = timeout
        result = predicate()
        while not result:
            if waittime is not None:
                if endtime is None:
                    endtime = _time() + waittime
                else:
                    waittime = endtime - _time()
                    if waittime <= 0:
                        break
            self.wait(waittime)
            result = predicate()
        return result

    def notify(self, n=1):
        if not self._is_owned():
            raise RuntimeError("cannot notify on un-acquired lock")
        all_waiters = self._waiters
        waiters_to_notify = _deque(_islice(all_waiters, n))
        if not waiters_to_notify:
            return
        for waiter in waiters_to_notify:
            waiter.release()
            try:
                all_waiters.remove(waiter)
            except ValueError:
                pass

    def notify_all(self):
        self.notify(len(self._waiters))

    notifyAll = notify_all


# A barrier class.  Inspired in part by the pthread_barrier_* api and
# the CyclicBarrier class from Java.  See
# http://sourceware.org/pthreads-win32/manual/pthread_barrier_init.html and
# http://java.sun.com/j2se/1.5.0/docs/api/java/util/concurrent/
#        CyclicBarrier.html
# for information.
# We maintain two main states, 'filling' and 'draining' enabling the barrier
# to be cyclic.  Threads are not allowed into it until it has fully drained
# since the previous cycle.  In addition, a 'resetting' state exists which is
# similar to 'draining' except that threads leave with a BrokenBarrierError,
# and a 'broken' state in which all threads get the exception.
class Barrier:
    """
    Barrier.  Useful for synchronizing a fixed number of threads
    at known synchronization points.  Threads block on 'wait()' and are
    simultaneously once they have all made that call.
    """
    def __init__(self, parties, action=None, timeout=None):
        """
        Create a barrier, initialised to 'parties' threads.
        'action' is a callable which, when supplied, will be called
        by one of the threads after they have all entered the
        barrier and just prior to releasing them all.
        If a 'timeout' is provided, it is uses as the default for
        all subsequent 'wait()' calls.
        """
        self._cond = Condition(Lock())
        self._action = action
        self._timeout = timeout
        self._parties = parties
        self._state = 0 #0 filling, 1, draining, -1 resetting, -2 broken
        self._count = 0

    def wait(self, timeout=None):
        """
        Wait for the barrier.  When the specified number of threads have
        started waiting, they are all simultaneously awoken. If an 'action'
        was provided for the barrier, one of the threads will have executed
        that callback prior to returning.
        Returns an individual index number from 0 to 'parties-1'.
        """
        if timeout is None:
            timeout = self._timeout
        with self._cond:
            self._enter() # Block while the barrier drains.
            index = self._count
            self._count += 1
            try:
                if index + 1 == self._parties:
                    # We release the barrier
                    self._release()
                else:
                    # We wait until someone releases us
                    self._wait(timeout)
                return index
            finally:
                self._count -= 1
                # Wake up any threads waiting for barrier to drain.
                self._exit()

    # Block until the barrier is ready for us, or raise an exception
    # if it is broken.
    def _enter(self):
        while self._state in (-1, 1):
            # It is draining or resetting, wait until done
            self._cond.wait()
        #see if the barrier is in a broken state
        if self._state < 0:
            raise BrokenBarrierError
        assert self._state == 0

    # Optionally run the 'action' and release the threads waiting
    # in the barrier.
    def _release(self):
        try:
            if self._action:
                self._action()
            # enter draining state
            self._state = 1
            self._cond.notify_all()
        except:
            #an exception during the _action handler.  Break and reraise
            self._break()
            raise

    # Wait in the barrier until we are relased.  Raise an exception
    # if the barrier is reset or broken.
    def _wait(self, timeout):
        if not self._cond.wait_for(lambda : self._state != 0, timeout):
            #timed out.  Break the barrier
            self._break()
            raise BrokenBarrierError
        if self._state < 0:
            raise BrokenBarrierError
        assert self._state == 1

    # If we are the last thread to exit the barrier, signal any threads
    # waiting for the barrier to drain.
    def _exit(self):
        if self._count == 0:
            if self._state in (-1, 1):
                #resetting or draining
                self._state = 0
                self._cond.notify_all()

    def reset(self):
        """
        Reset the barrier to the initial state.
        Any threads currently waiting will get the BrokenBarrier exception
        raised.
        """
        with self._cond:
            if self._count > 0:
                if self._state == 0:
                    #reset the barrier, waking up threads
                    self._state = -1
                elif self._state == -2:
                    #was broken, set it to reset state
                    #which clears when the last thread exits
                    self._state = -1
            else:
                self._state = 0
            self._cond.notify_all()

    def abort(self):
        """
        Place the barrier into a 'broken' state.
        Useful in case of error.  Any currently waiting threads and
        threads attempting to 'wait()' will have BrokenBarrierError
        raised.
        """
        with self._cond:
            self._break()

    def _break(self):
        # An internal error was detected.  The barrier is set to
        # a broken state all parties awakened.
        self._state = -2
        self._cond.notify_all()

    @property
    def parties(self):
        """
        Return the number of threads required to trip the barrier.
        """
        return self._parties

    @property
    def n_waiting(self):
        """
        Return the number of threads that are currently waiting at the barrier.
        """
        # We don't need synchronization here since this is an ephemeral result
        # anyway.  It returns the correct value in the steady state.
        if self._state == 0:
            return self._count
        return 0

    @property
    def broken(self):
        """
        Return True if the barrier is in a broken state
        """
        return self._state == -2


#exception raised by the Barrier class
class BrokenBarrierError(RuntimeError): pass



"""
From: https://github.com/mxbi/mlcrate
"""


class SuperPool:
    def __init__(self, n_cpu=-1):
        """Process pool for applying functions multi-threaded with progress bars.
        Arguments:
        n_cpu -- The number of processes to spawn. Defaults to the number of threads (logical cores) on your system.
        Usage:
        >>> pool = mlc.SuperPool()  # By default, the cpu count is used
        >>> def f(x):
        ...     return x ** 2
        >>> res = pool.map(f, range(1000))  # Apply function f to every value in y
        [mlcrate] 8 CPUs: 100%|████████████████████████| 1000/1000 [00:00<00:00, 1183.78it/s]
        """
        from multiprocessing import cpu_count
        from pathos.multiprocessing import ProcessPool
        import tqdm

        self.tqdm = tqdm

        if n_cpu == -1:
            n_cpu = cpu_count()

        self.n_cpu = n_cpu
        self.pool = ProcessPool(n_cpu)

    def __del__(self):
        self.pool.close()

    def map(self, func, array, chunksize=16, description=""):
        """Map a function over array using the pool and return [func(a) for a in array].
        Arguments:
        func -- The function to apply. Can be a lambda function
        array -- Any iterable to which the function should be applied over
        chunksize (default: 16) -- The size of a "chunk" which is sent to a CPU core for processing in one go. Larger values should speed up processing when using very fast functions, while smaller values will give a more granular progressbar.
        description (optional) -- Text to be displayed next to the progressbar.
        Returns:
        res -- A list of values returned from the function.
        """
        res = []

        def func_tracked(args):
            x, i = args
            return func(x), i

        array_tracked = zip(array, range(len(array)))

        desc = "{} CPUs{}".format(
            self.n_cpu, " - {}".format(description) if description else ""
        )
        for out in self.tqdm.tqdm(
            self.pool.uimap(func_tracked, array_tracked, chunksize=chunksize),
            total=len(array),
            desc=desc,
            smoothing=0.05,
        ):
            res.append(out)

        # Sort based on i but return only the actual function result
        actual_res = [r[0] for r in sorted(res, key=lambda r: r[1])]

        return actual_res

    def exit(self):
        """Close the processes and wait for them to clean up."""
        self.pool.close()
        self.pool.join()

import typing as tp
from . import operations as ops
from . import external_sort as ext_sort


class Graph:
    """Computational graph implementation"""

    def __init__(self, operation: ops.Operation, parents: list['Graph']) -> None:
        self._operation: ops.Operation = operation
        self._parents: list[Graph] = parents

    def copy(self) -> 'Graph':
        """Copies the graph"""
        return Graph(self._operation, self._parents)

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterGenerator
        :param name: name of kwarg to use as data source
        """
        return Graph(ops.ReadIterGenerator(name), [])

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        return Graph(ops.Read(filename, parser), [])

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        return Graph(ops.Map(mapper), [self])

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        return Graph(ops.Reduce(reducer, keys), [self])

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        return Graph(ext_sort.ExternalSort(keys), [self])

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        return Graph(ops.Join(joiner, keys), [self, join_graph])

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        parents_run: list[ops.TRowsIterable] = [parent.run(**kwargs) for parent in self._parents]
        yield from self._operation(*parents_run, **kwargs)

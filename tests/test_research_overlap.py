import pytest
from research.plot_trace import union, overlap_seconds


def test_nested_cpu_intervals_are_not_counted_twice():
    assert overlap_seconds([(0,10),(2,8),(4,7)],[(3,6)]) == 3


def test_adjacent_half_open_intervals_have_no_positive_overlap():
    assert overlap_seconds([(0,2)],[(2,5)]) == 0
    assert union([(0,2),(2,3)]) == [(0,3)]


def test_disjoint_gpu_intervals_and_multiple_cpu_intersections():
    assert overlap_seconds([(1,9),(5,11)],[(0,3),(7,10)]) == 5


@pytest.mark.parametrize('bad', [[(3,2)],[(float('nan'),2)],[(0,float('inf'))],[(False,2)]])
def test_invalid_interval_does_not_become_a_measurement(bad):
    with pytest.raises(ValueError):
        overlap_seconds(bad,[(0,5)])

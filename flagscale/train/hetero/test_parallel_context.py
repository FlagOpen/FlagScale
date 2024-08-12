from flagscale.train.hetero.parallel_context import find_overlapped_mapping, RankMapper, ProcessMesh, ParallelContext
# from megatron import get_args

# args = get_args()

def test_find_overlapped_mapping():
    dim1, dim2 = 3, 4
    result = {
        0: [(0,0,3), (1,3,4)],
        1: [(1,0,2), (2,2,4)],
        2: [(2,0,1), (3,1,4)]
    }
    assert find_overlapped_mapping(dim1=dim1, dim2=dim2) == result
    dim1, dim2 = 2, 4
    result = {
        0: [(0,0,1), (1,1,2)],
        1: [(2,0,1), (3,1,2)]
    }
    assert find_overlapped_mapping(dim1=dim1, dim2=dim2) == result
    dim1, dim2 = 3, 2
    result = {
        0: [(0,0,2)],
        1: [(0,0,1), (1,1,2)],
        2: [(1,0,2)]
    }
    assert find_overlapped_mapping(dim1=dim1, dim2=dim2) == result

# def test_RankMapper():
#     RankMapper()
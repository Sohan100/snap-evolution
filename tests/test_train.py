from snapevo.train import _stalled


def test_ladder_stops_when_the_floor_stalls():
    r = [{"D": 3, "C_train": 1e-3}, {"D": 5, "C_train": 1e-4}, {"D": 7, "C_train": 6e-5}, {"D": 9, "C_train": 6e-5}]
    assert not _stalled(r, 5)        # nothing 4 or more sublayers below D = 5
    assert not _stalled(r, 7)        # 16x below D = 3
    assert _stalled(r, 9)            # less than 2x below D = 5

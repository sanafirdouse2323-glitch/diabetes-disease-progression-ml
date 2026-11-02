from diabetes_ml.data import load_diabetes_frame, make_train_test_split


def test_load_diabetes_frame_shape_and_alignment():
    X, y = load_diabetes_frame()
    assert X.shape == (442, 10)
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0


def test_split_is_reproducible_and_disjoint():
    X, y = load_diabetes_frame()
    first = make_train_test_split(X, y, random_state=7)
    second = make_train_test_split(X, y, random_state=7)
    assert first.X_train.index.equals(second.X_train.index)
    assert first.X_test.index.equals(second.X_test.index)
    assert set(first.X_train.index).isdisjoint(set(first.X_test.index))

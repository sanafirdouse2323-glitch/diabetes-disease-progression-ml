from diabetes_ml.data import load_diabetes_frame, make_train_test_split
from diabetes_ml.models import build_models


def test_every_model_fits_and_predicts():
    X, y = load_diabetes_frame()
    split = make_train_test_split(X, y)
    for model in build_models().values():
        model.fit(split.X_train, split.y_train)
        predictions = model.predict(split.X_test)
        assert len(predictions) == len(split.y_test)

import joblib
import pandas as pd
from flask import url_for


def ia_result(specie, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    # renvoie le sex predit par une methode IA

    IA_test = joblib.load(
        "pingouin_app" + url_for("static", filename="model_IA/reg_log_pingouin.pkl")
    )
    scaler = joblib.load(
        "pingouin_app" + url_for("static", filename="model_IA/label/scaler.pkl")
    )
    label_sex = joblib.load(
        "pingouin_app" + url_for("static", filename="model_IA/label/label_sex.pkl")
    )
    label_specie = joblib.load(
        "pingouin_app" + url_for("static", filename="model_IA/label/label_species.pkl")
    )

    # convertion des données dans le bon format
    bill_length_mm = float(bill_length_mm)
    bill_depth_mm = float(bill_depth_mm)
    flipper_length_mm = float(flipper_length_mm)
    body_mass_g = float(body_mass_g)

    # creation d'un mini dataframe
    dico_pingouin = {
        "species": specie,
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
    }
    test = pd.DataFrame(data=dico_pingouin, index=["predict"])

    # modification du dataframe pour correspondre à l'entrainement de l'IA
    test[
        ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    ] = scaler.transform(
        test[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
    )
    test["species"] = label_specie.transform(test["species"])

    # retournement du sexe determiner par IA
    sex_number = IA_test.predict(test)
    sex_predict = label_sex.inverse_transform(sex_number)
    sex_score = IA_test.predict_proba(test)

    return (sex_predict[0], sex_score[0, sex_number[0]])

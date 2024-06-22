from flask_wtf import FlaskForm
from wtforms import SelectField, DecimalField
from wtforms.validators import DataRequired, NumberRange


class Form_ia_pingouin(FlaskForm):
    specie = SelectField(
        "espèce", validators=[DataRequired()], choices=["Adelie", "Gentoo", "Chinstrap"]
    )
    bill_length_mm = DecimalField(
        "longueur du nez", validators=[NumberRange(20, 70), DataRequired()]
    )
    bill_depth_mm = DecimalField(
        "grandeur du nez", validators=[NumberRange(10, 25), DataRequired()]
    )
    flipper_length_mm = DecimalField(
        "Longueur des ailes", validators=[NumberRange(150, 250), DataRequired()]
    )
    body_mass_g = DecimalField(
        "Poids du pingouin en gramme",
        validators=[NumberRange(min=2000, max=7000), DataRequired()],
    )

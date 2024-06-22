# import bibliothèque flask et création application flask
from flask import render_template, request, current_app, send_from_directory
from .form import Form_ia_pingouin
from .util import ia_result

app = current_app


@app.route("/robots.txt")
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


@app.route("/", methods=["GET", "POST"])
@app.route("/index/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/cv/", methods=["GET", "POST"])
def cv():
    return render_template("CV.html")


@app.route("/data/", methods=["GET", "POST"])
def data():
    return render_template("data/data.html")


@app.route("/determination_pingouin/", methods=["GET", "POST"])
def form_ia_pingouin():
    form = Form_ia_pingouin()
    return render_template("data/form_ia_penguin.html", form=form)


@app.route("/resultat_pingouin/", methods=["POST"])
def resultat_pingouin():
    specie = request.form["specie"]
    bill_length_mm = request.form["bill_length_mm"]
    bill_depth_mm = request.form["bill_depth_mm"]
    flipper_length_mm = request.form["flipper_length_mm"]
    body_mass_g = request.form["body_mass_g"]
    sex_prediction, sex_score = ia_result(
        specie, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
    )
    sex_score = round(sex_score * 100, 2)

    return render_template(
        "data/resultat_pingouin.html",
        specie=specie,
        bill_length_mm=bill_length_mm,
        bill_depth_mm=bill_depth_mm,
        flipper_length_mm=flipper_length_mm,
        body_mass_g=body_mass_g,
        sex_prediction=sex_prediction,
        sex_score=sex_score,
    )


@app.route("/description_IA_pingouin/", methods=["GET", "POST"])
def description_ia_pingouin():
    return render_template("data/description_ia_penguin.html")


@app.route("/python_data/", methods=["GET", "POST"])
def python_data():
    return render_template("data/python_data.html")


@app.route("/python_graph/", methods=["GET", "POST"])
def python_graph():
    return render_template("data/python_graph.html")


@app.route("/aide_memoire/", methods=["GET", "POST"])
def aide_memoire():
    return render_template("aide_memoire/aide_memoire.html")


@app.route("/git/", methods=["GET", "POST"])
def git():
    return render_template("aide_memoire/aide_memoire_git.html")


@app.route("/git_action/", methods=["GET", "POST"])
def git_action():
    return render_template("aide_memoire/aide_memoire_git_action.html")


@app.route("/ansible/", methods=["GET", "POST"])
def ansible():
    return render_template("aide_memoire/aide_memoire_ansible.html")


@app.route("/docker/", methods=["GET", "POST"])
def tip_docker():
    return render_template("aide_memoire/aide_memoire_docker.html")


@app.route("/flask/", methods=["GET", "POST"])
def tip_flask():
    return render_template("aide_memoire/aide_memoire_flask.html")


@app.route("/variable_serveur/", methods=["GET", "POST"])
def tips_variable_systeme():
    return render_template("aide_memoire/aide_memoire_variable_systeme.html")


@app.route("/terraform/", methods=["GET", "POST"])
def terraform():
    return render_template("aide_memoire/aide_memoire_terraform.html")


@app.route("/aws_cdk/", methods=["GET", "POST"])
def aws_cdk():
    return render_template("aide_memoire/aide_memoire_aws_cdk.html")


@app.route("/liens/", methods=["GET", "POST"])
def lien():
    return render_template("liens_utile.html")

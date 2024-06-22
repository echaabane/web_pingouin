# teste si l'application envoie bien les pages avec la requete get


def test_index_get_status_codeOK(client):
    response = client.get("/index/")
    assert response.status_code == 200


def test_origin_get_status_codeOK(client):
    response = client.get("/")
    assert response.status_code == 200


def test_CV_get_status_codeOK(client):
    response = client.get("/cv/")
    assert response.status_code == 200


def test_data_get_status_codeOK(client):
    response = client.get("/data/")
    assert response.status_code == 200


def test_description_IA_pingouin_get_status_codeOK(client):
    response = client.get("/description_IA_pingouin/")
    assert response.status_code == 200


def test_python_data_get_status_codeOK(client):
    response = client.get("/python_data/")
    assert response.status_code == 200


def test_python_graph_get_status_codeOK(client):
    response = client.get("/python_graph/")
    assert response.status_code == 200


def test_aide_memoire_get_status_codeOK(client):
    response = client.get("/aide_memoire/")
    assert response.status_code == 200


def test_git_get_status_codeOK(client):
    response = client.get("/git/")
    assert response.status_code == 200


def test_git_action_get_status_codeOK(client):
    response = client.get("/git_action/")
    assert response.status_code == 200


def test_ansible_get_status_codeOK(client):
    response = client.get("/ansible/")
    assert response.status_code == 200


def test_docker_get_status_codeOK(client):
    response = client.get("/docker/")
    assert response.status_code == 200


def test_flask_get_status_codeOK(client):
    response = client.get("/flask/")
    assert response.status_code == 200


def test_variable_serveur_get_status_codeOK(client):
    response = client.get("/variable_serveur/")
    assert response.status_code == 200


def test_terraform_get_status_codeOK(client):
    response = client.get("/terraform/")
    assert response.status_code == 200


def test_liens_get_status_codeOK(client):
    response = client.get("/liens/")
    assert response.status_code == 200


# test avec la requete POST
def test_index_post_status_codeOK(client):
    response = client.post("/index/")
    assert response.status_code == 200


def test_origin_post_status_codeOK(client):
    response = client.post("/")
    assert response.status_code == 200


def test_CV_post_status_codeOK(client):
    response = client.post("/cv/")
    assert response.status_code == 200


def test_data_post_status_codeOK(client):
    response = client.post("/data/")
    assert response.status_code == 200


def test_description_IA_pingouin_post_status_codeOK(client):
    response = client.post("/description_IA_pingouin/")
    assert response.status_code == 200


def test_python_data_post_status_codeOK(client):
    response = client.post("/python_data/")
    assert response.status_code == 200


def test_python_graph_post_status_codeOK(client):
    response = client.post("/python_graph/")
    assert response.status_code == 200


def test_aide_memoire_post_status_codeOK(client):
    response = client.post("/aide_memoire/")
    assert response.status_code == 200


def test_git_post_status_codeOK(client):
    response = client.post("/git/")
    assert response.status_code == 200


def test_git_action_post_status_codeOK(client):
    response = client.post("/git_action/")
    assert response.status_code == 200


def test_ansible_post_status_codeOK(client):
    response = client.post("/ansible/")
    assert response.status_code == 200


def test_docker_post_status_codeOK(client):
    response = client.post("/docker/")
    assert response.status_code == 200


def test_flask_post_status_codeOK(client):
    response = client.post("/flask/")
    assert response.status_code == 200


def test_variable_serveur_post_status_codeOK(client):
    response = client.post("/variable_serveur/")
    assert response.status_code == 200


def test_terraform_post_status_codeOK(client):
    response = client.post("/terraform/")
    assert response.status_code == 200


def test_liens_post_status_codeOK(client):
    response = client.post("/liens/")
    assert response.status_code == 200


# voir pour integrer des tests avec post
def test_determination_pingouin_status_codeOK(client):
    with client:
        response = client.post(
            "/resultat_pingouin/",
            data={
                "specie": "Gentoo",
                "bill_length_mm": 40,
                "bill_depth_mm": 17,
                "flipper_length_mm": 200,
                "body_mass_g": 4500,
            },
        )
        assert response.status_code == 200

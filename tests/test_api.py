import requests

def test_send_image():
    file = {'file': open('lama_300px.png', 'rb')}
    resp = requests.post('http://127.0.0.1:5000/upscale/', files=file)
    assert resp.status_code == 200
    resp_data = resp.json()
    assert resp_data['task_id']
    global test_task_id
    test_task_id = resp_data['task_id']
    assert resp_data['filename'].split('.')[0][-3:] == 'out'


def test_get_status():
    resp = requests.get(f'http://127.0.0.1:5000/tasks/{test_task_id}')
    resp_data = resp.json()
    assert resp.status_code == 200
    assert resp_data == {'result': None, 'status': 'PENDING'}
    assert resp_data['status']
    assert 'result' in resp_data


def test_get_status_wrong_task_id():
    wrong_task_id = 'wrong_task_id'
    resp = requests.get(f'http://127.0.0.1:5000/tasks/{wrong_task_id}')
    assert resp.status_code == 400
    resp_data = resp.json()
    assert resp_data == {'status': 'error', 'message': 'wrong task_id'}



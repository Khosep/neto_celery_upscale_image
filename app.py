import uuid
import os
import cv2
from flask import Flask, request, jsonify, send_file, send_from_directory, redirect, url_for, Response, abort
from flask.views import MethodView
from celery import Celery
from celery.result import AsyncResult
from cv2 import dnn_superres
# from upscale import upscale


APP_NAME = 'app'
app = Flask(APP_NAME)
app.config['UPLOAD_FOLDER'] = 'file'

def get_celery_app_instance(app=None):
    celery = Celery(APP_NAME, backend='redis://localhost:6379/1', broker='redis://localhost:6379/2')
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = get_celery_app_instance(app)


@celery.task()
def upscale(input_path: str, output_path: str, model_path: str = 'EDSR_x2.pb'):
    """
    :param input_path: путь к изображению для апскейла
    :param output_path:  путь к выходному файлу
    :param model_path: путь к ИИ модели
    :return:
    """
    scaler = dnn_superres.DnnSuperResImpl_create()
    scaler.readModel(model_path)
    scaler.setModel("edsr", 2)
    image = cv2.imread(input_path)
    result = scaler.upsample(image)
    cv2.imwrite(output_path, result)
    # print(all([image.shape[i] * 2 == result.shape[i] for i in range(2)]))
    return output_path

def get_filename(path: str) -> str:
    return path[len(app.config['UPLOAD_FOLDER']) + 1:]


class UpscaleView(MethodView):
    def get(self, task_id):
        task = AsyncResult(task_id, app=celery)
        active_tasks = list(celery.control.inspect().active().values())[0]
        if any(a_task['id'] == task_id for a_task in active_tasks):
            return jsonify({'status': task.status,
                            'result': task.result,
                            })
        elif task.result:
            filename = get_filename(task.result)
            return redirect(url_for('processed', filename=filename), 303)
        else:
            return jsonify({'status': 'error', 'message': 'wrong task_id'}), 400


    def post(self):
        path_in, path_out = self.save_image()
        task = upscale.delay(path_in, path_out)
        return jsonify({'task_id': task.id,
                        'filename': get_filename(path_out)})

    def save_image(self):
        image = request.files.get('file')
        extension = image.filename.split('.')[-1]
        name = f'{uuid.uuid4()}'
        path_in = os.path.join('file', f'{name}_in.{extension}')
        path_out = os.path.join('file', f'{name}_out.{extension}')
        image.save(path_in)
        return path_in, path_out


class ProcessedView(MethodView):
    def get(self, filename):
        return send_file(path_or_file=os.path.join(app.config['UPLOAD_FOLDER'], filename))


upscale_view = UpscaleView.as_view('upscale')
processed_view = ProcessedView.as_view('processed')

app.add_url_rule('/tasks/<string:task_id>', view_func=upscale_view, methods=['GET'])
app.add_url_rule('/upscale/', view_func=upscale_view, methods=['POST'])
app.add_url_rule('/processed/<path:filename>', view_func=processed_view, methods=['GET'])

if __name__ == '__main__':
    app.run()

# display.py
import json
import os
import math
from functools import lru_cache
from flask import Flask, render_template, send_from_directory, abort, request, url_for
from urllib.parse import quote, unquote
import webbrowser
import argparse  # 新增导入

# 解析命令行参数
parser = argparse.ArgumentParser(description='Display image information using Flask app.')
parser.add_argument('--per_page', type=int, default=8, help='Number of items per page.')
parser.add_argument('--input_json', type=str, default='face.json', help='Input JSON file path for detection results.')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the Flask app on.')
parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on.')
parser.add_argument('--debug', action='store_true', help='Run the Flask app in debug mode.')
args = parser.parse_args()

app = Flask(__name__)
app.config.update({
    'PER_PAGE': args.per_page,  # 使用命令行参数
    'JSON_PATH': os.path.normpath(os.path.join(os.path.dirname(__file__), args.input_json))  # 使用命令行参数
})

app.jinja_env.filters['urlencode'] = lambda s: quote(s.encode('utf-8'))

class Paginator:
    @staticmethod
    def paginate(items, page, per_page):
        total = len(items)
        if total == 0:
            return [], 1
        total_pages = math.ceil(total / per_page)
        page = max(1, min(page, total_pages))
        start = (page - 1) * per_page
        return items[start:start+per_page], total_pages

@lru_cache(maxsize=1)
def load_image_data():
    try:
        with open(app.config['JSON_PATH'], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        app.logger.error(f"Failed to load image data: {str(e)}")
        return {}

def process_image_metadata():
    data = load_image_data()
    category_map = {}
    file_map = {}
    
    for raw_path, meta in data.items():
        if not meta.get('face_scores'):
            continue
            
        normalized = os.path.normpath(raw_path)
        dir_path, filename = os.path.split(normalized)
        category = os.path.basename(dir_path)
        
        if not category:
            continue
            
        img_info = {
            'filename': filename,
            'category': category,
            'path': normalized,
            'face_scores': meta.get('face_scores', []),
            'landmark_scores': meta.get('face_landmark_scores_68', [])
        }
        
        category_map.setdefault(category, []).append(img_info)
        file_map[f"{category}/{filename}"] = normalized
    
    return category_map, file_map

def get_category_thumbnail(category_map, category):
    return category_map.get(category, [{}])[0]

@app.route('/image/<category>/<path:filename>')
def serve_image(category, filename):
    _, file_map = process_image_metadata()
    unique_id = f"{unquote(category)}/{filename}"
    
    if unique_id not in file_map:
        abort(404, description="Image not found")
    
    return send_from_directory(
        os.path.dirname(file_map[unique_id]),
        os.path.basename(file_map[unique_id])
    )

def render_category_view(page, category=None):
    category_map, _ = process_image_metadata()
    sorted_categories = sorted(category_map.keys())
    
    if category:
        items = category_map.get(category, [])
    else:
        items = [img for cat in sorted_categories for img in category_map.get(cat, [])]
    
    paginated, total_pages = Paginator.paginate(items, page, app.config['PER_PAGE'])
    
    return render_template('index.html',
                        images=paginated,
                        current_page=page,
                        total_pages=total_pages,
                        category=category,
                        all_categories=sorted_categories)

@app.route('/')
def show_categories():
    page = request.args.get('page', 1, type=int)
    category_map, _ = process_image_metadata()
    sorted_categories = sorted(category_map.keys())
    
    categories, total_pages = Paginator.paginate(
        sorted_categories, page, app.config['PER_PAGE'])
    
    category_list = [{
        'name': cat,
        'thumb_url': url_for('serve_image', 
                            category=cat, 
                            filename=get_category_thumbnail(category_map, cat).get('filename', '')),
        'url': url_for('category_view', category=cat, page=1)
    } for cat in categories]
    
    return render_template('categories.html',
                         categories=category_list,
                         current_page=page,
                         total_pages=total_pages)

@app.route('/all')
def show_all_images():
    page = request.args.get('page', 1, type=int)
    return render_category_view(page)

@app.route('/<path:category>/<int:page>')
@app.route('/<path:category>', defaults={'page': 1})
def category_view(category, page):
    try:
        current_category = unquote(category)
    except UnicodeDecodeError:
        abort(404, description="Invalid category name")
    
    return render_category_view(page, current_category)

@app.errorhandler(404)
def handle_404(e):
    return render_template('404.html', message=e.description), 404

if __name__ == '__main__':
    import threading

    def input_thread():
        print('Press q to quit')
        while True:
            user_input = input()
            if user_input.lower() == 'q':
                # 这里添加关闭 Flask 应用的逻辑
                # 比如可以使用一个全局变量来控制应用的运行状态
                # 这里简单地调用 os._exit 强制退出
                import os
                os._exit(0)

    # 启动输入监听线程
    input_t = threading.Thread(target=input_thread)
    input_t.daemon = True
    input_t.start()

    def open_browser():
        webbrowser.open_new(f'http://127.0.0.1:{args.port}')

    # 使用线程来打开浏览器，避免阻塞Flask应用启动
    threading.Timer(1, open_browser).start() 

    app.run(host=args.host, port=args.port, debug=args.debug)
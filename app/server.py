import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

#export_file_url = 'https://www.dropbox.com/s/2mpmxuzh5a85ms8/exportFlowers.pkl?dl=1'    #Initial version
export_file_url = 'https://www.dropbox.com/s/0uxyitiozbjxx3e/exportBigFlowers.pkl?dl=1'  #Big training set version
export_file_name = 'exportBigFlowers.pkl'

classes = ['alpine_sea_holly',
 'anthurium',
 'artichoke',
 'azalea',
 'ball_moss',
 'balloon_flower',
 'barbeton_daisy',
 'bearded_iris',
 'bee_balm',
 'bird_of_paradise',
 'bishop_of_llandaff',
 'black-eyed_susan',
 'blackberry_lily',
 'blanket_flower',
 'bolero_deep_blue',
 'bougainvillea',
 'bromelia',
 'buttercup',
 'californian_poppy',
 'camellia',
 'canna_lily',
 'canterbury_bells',
 'cape_flower',
 'carnation',
 'cautleya_spicata',
 'clematis',
 "colt's_foot",
 'columbine',
 'common_dandelion',
 'corn_poppy',
 'cyclamen_',
 'daffodil',
 'desert-rose',
 'english_marigold',
 'fire_lily',
 'foxglove',
 'frangipani',
 'fritillary',
 'garden_phlox',
 'gaura',
 'gazania',
 'geranium',
 'giant_white_arum_lily',
 'globe-flower',
 'globe_thistle',
 'grape_hyacinth',
 'great_masterwort',
 'hard-leaved_pocket_orchid',
 'hibiscus',
 'hippeastrum_',
 'japanese_anemone',
 'king_protea',
 'lenten_rose',
 'lotus',
 'love_in_the_mist',
 'magnolia',
 'mallow',
 'marigold',
 'mexican_aster',
 'mexican_petunia',
 'monkshood',
 'moon_orchid',
 'morning_glory',
 'orange_dahlia',
 'osteospermum',
 'oxeye_daisy',
 'passion_flower',
 'pelargonium',
 'peruvian_lily',
 'petunia',
 'pincushion_flower',
 'pink-yellow_dahlia?',
 'pink_primrose',
 'poinsettia',
 'primula',
 'prince_of_wales_feathers',
 'purple_coneflower',
 'red_ginger',
 'rose',
 'ruby-lipped_cattleya',
 'siam_tulip',
 'silverbush',
 'snapdragon',
 'spear_thistle',
 'spring_crocus',
 'stemless_gentian',
 'sunflower',
 'sweet_pea',
 'sweet_william',
 'sword_lily',
 'thorn_apple',
 'tiger_lily',
 'toad_lily',
 'tree_mallow',
 'tree_poppy',
 'trumpet_creeper',
 'wallflower',
 'water_lily',
 'watercress',
 'wild_pansy',
 'windflower',
 'yellow_iris']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)
    top3 = sorted(zip(np.array(prediction[2])*100, classes), reverse=True)[:3]
    result = ', '.join([f'{p:.0f}% {g.replace("_"," ")}' for p,g in top3])
    return JSONResponse({'result': result})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

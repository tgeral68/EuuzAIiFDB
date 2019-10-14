import sys
import tqdm
from urllib import request
from os.path import join

# donwload ressource at url 
# save at the location given by filepath
def download(url, filepath, verbose=2):
    file_name = url.split('/')[-1]
    url_obj = request.urlopen(url)
    meta = url_obj.info()
    f = open(filepath, 'wb')
    file_size_dl = 0
    block_sz = 8192

    try:
        file_size = int(meta.get("Content-Length"))
        # define size of donwloaded blocks

        if(verbose>=1):
            print("\n\nDownloading: "+file_name+" "+str(file_size/(1000*1000)//1)+" Mo")
            sys.stdout.flush()
            pb = tqdm.trange(int(file_size/block_sz)+1)
        nb_package = 0
        while True:
            buffer = url_obj.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            nb_package += 1
            if(verbose>=1):
                pb.update(1)
                pb.set_postfix({"Downloaded": str(((nb_package*block_sz)/(1000*1000))*10//1/10)+" Mo"})
    except Exception:
        if(verbose>=1):
            print("\n\nDownloading: "+file_name+" (size unvailable)")
        while True:
            buffer = url_obj.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
    if(verbose>=1):
        print("\n")
    f.close()
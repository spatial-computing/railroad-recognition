import argparse
import os
import shutil
import zipfile
import urllib2
from IPython import embed


# How to run:
# python download_raster_map.py location_url
# By default will download and extract zip file to ./raw_data

# https://prd-tnm.s3.amazonaws.com/hm_archive/CA/CA_Bray_100414_2001_24000_bag.zip


def download_raster_map(download_url, working_dir="../raw_data"):
    if 'https:' in download_url:
        filename = download_url.split('/')[-1]
        outfile = os.path.join(working_dir, filename)

        try:
            if os.path.exists(outfile):
                print 'File already exists', outfile

            else:
                request = urllib2.urlopen(download_url.replace(' ', '%20'))

                with open(outfile, 'wb') as fp:
                    shutil.copyfileobj(request, fp)

                print download_url, 'OK'
                del fp

        except Exception as e:
            print 'ERROR downloading', download_url, e.message

        # Unzip file
        print "Finished downloading"
        print "Unzipping.."
        extract_to = outfile.replace(".zip", "")

        if not os.path.exists(extract_to):
            zip_ref = zipfile.ZipFile(outfile, 'r')
            zip_ref.extractall(working_dir)
            zip_ref.close()

        return extract_to


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unzip rasterized map')
    parser.add_argument("url", help='url of the map')

    args = parser.parse_args()
    url = args.url

    download_raster_map(url)

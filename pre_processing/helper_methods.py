import subprocess

gdalsrsinfo_command = r'gdalsrsinfo'
gdal_translate_command = r'gdal_translate'
gdalwarp_command = r'gdalwarp'
ogr2ogr_command = r'ogr2ogr'


def run_and_return_output(sys_call):
    print "Running " + sys_call

    try:
        return subprocess.check_output(sys_call, stderr=subprocess.STDOUT, shell=True).decode().strip().replace("'", "")
    except subprocess.CalledProcessError, e:
        message = "Command '%s' returned non-zero exit status %d with the following message:\n %s" \
                  % (e.cmd, e.returncode, e.output)
        raise RuntimeError(message)


def gdalsrsinfo(*args):
    return run_and_return_output(gdalsrsinfo_command + " " + " ".join(args))


def gdal_translate(*args):
    return run_and_return_output(gdal_translate_command + " " + " ".join(args))


def gdalwarp(*args):
    return run_and_return_output(gdalwarp_command + " " + " ".join(args))


def ogr2ogr(*args):
    return run_and_return_output(ogr2ogr_command + " " + " ".join(args))

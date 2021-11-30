from pathlib import Path

from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from webdav3.client import Client

logger = defaultLogger("deepfashion_environment")

class WebDav:
    def __init__(self, base_path, **webdav_options):
        """
        Required Params: base
        :param base_path: Base Path of where to move files to e.q. /users/name/project_x/
        :param webdav_options: required kwargs: 'webdav_hostname', 'webdav_login', 'webdav_password'
        """

        self.webdav_options = webdav_options
        self.client = self.login()

        self.base_path = base_path

        if not self.base_path.endswith("/"):
            self.base_path += "/"

    def login(self):
        return Client(self.webdav_options)

    def upload(self, src_local, dst_remote, callback, _async=True):
        kwargs = {
            'remote_path': dst_remote,
            'local_path': src_local,
            'callback': callback
        }

        if _async:
            self.client.upload_async(**kwargs)
        else:
            self.client.upload_sync(**kwargs)

    def download(self, src_remote, dst_local, callback, _async=False):
        kwargs = {
            'remote_path': src_remote,
            'local_path': dst_local,
            'callback': callback
        }

        if _async:
            self.client.download_async(**kwargs)
        else:
            self.client.download_sync(**kwargs)

        return dst_local

    @staticmethod
    def remove_local(path, callback=None, ignore_exception=True):
        try:
            Path(path).unlink()
            callback()
            logger.info(f"Uploading: {path} done.")
        except Exception as e:
            if not ignore_exception:
                raise e

    def move(self, src_local, _async=True):
        f_name = Path(src_local).name
        dst_remote = self.base_path + f_name

        self.client.mkdir(self.base_path)

        callback = lambda: WebDav.remove_local(src_local, lambda: logger.info(f"moved {src_local} to {dst_remote}"))
        logger.info(f"Uploading: {src_local}")
        self.upload(src_local, dst_remote, callback, _async)

    def list(self, path, _filter=None):
        if not self.client.check(path):
            return []

        files = self.client.list(path)

        if _filter:
            files = list(filter(_filter, files))

        return files

    def __str__(self):
        return f"fashionnets.util.remote.WebDav{{base_path = {self.base_path}}}"



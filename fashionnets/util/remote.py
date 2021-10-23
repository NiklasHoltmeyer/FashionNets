from pathlib import Path

from webdav3.client import Client

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

    def upload(self, src_local, dst_remote, callback):
        kwargs = {
            'remote_path': dst_remote,
            'local_path': src_local,
            'callback': callback
        }

        self.client.upload_async(**kwargs)

    @staticmethod
    def remove_local(path, callback=None, ignore_exception=True):
        try:
            Path(path).unlink()
            callback()
            print(f"Uploading: {path} done.")
        except Exception as e:
            if not ignore_exception:
                raise e

    def move(self, src_local):
        f_name = Path(src_local).name
        dst_remote = self.base_path + f_name

        self.client.mkdir(self.base_path)

        callback = lambda: WebDav.remove_local(src_local, lambda: print(f"moved {src_local} to {dst_remote}"))
        print(f"Uploading: {src_local}")
        self.upload(src_local, dst_remote, callback)
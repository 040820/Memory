"""Configure the Qwen vLLM non-thinking mode inside the experiment process."""
import os


def install_non_thinking_adapter(manager_class, target_model):
    original = manager_class._generate_internal

    async def wrapped(self, prompt, **kwargs):
        if self.model == target_model:
            body = dict(kwargs.get('extra_body') or {})
            template = dict(body.get('chat_template_kwargs') or {})
            template['enable_thinking'] = False
            body['chat_template_kwargs'] = template
            kwargs['extra_body'] = body
        return await original(self, prompt, **kwargs)

    manager_class._generate_internal = wrapped


def install_local_client_proxy_guard(manager_class):
    """Construct local LLM clients without inheriting SOCKS proxy settings.

    Launchers must install this before creating managers and initialize them
    serially. Existing environment values are restored even if setup fails.
    """
    original = manager_class.__init__
    keys = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')

    def initialize(self, *args, **kwargs):
        saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
        try:
            return original(self, *args, **kwargs)
        finally:
            os.environ.update(saved)
    manager_class.__init__ = initialize

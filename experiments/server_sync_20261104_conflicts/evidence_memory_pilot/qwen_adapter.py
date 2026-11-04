"""Configure the Qwen vLLM non-thinking mode inside the experiment process."""


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

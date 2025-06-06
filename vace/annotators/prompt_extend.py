# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch

class PromptExtendAnnotator:
    def __init__(self, cfg, device=None):
        from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
        self.mode = cfg.get('MODE', "local_qwen")
        self.model_name = cfg.get('MODEL_NAME', "Qwen2.5_3B")
        self.is_vl = cfg.get('IS_VL', False)
        self.system_prompt = cfg.get('SYSTEM_PROMPT', None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else (torch.device(device) if isinstance(device, str) else device)
        self.device_id = self.device.index if self.device.type == 'cuda' else None
        rank = self.device_id if self.device_id is not None else 0
        if self.mode == "dashscope":
            self.prompt_expander = DashScopePromptExpander(
                model_name=self.model_name, is_vl=self.is_vl)
        elif self.mode == "local_qwen":
            self.prompt_expander = QwenPromptExpander(
                model_name=self.model_name,
                is_vl=self.is_vl,
                device=rank)
        else:
            raise NotImplementedError(f"Unsupport prompt_extend_method: {self.mode}")


    def forward(self, prompt, system_prompt=None, seed=-1):
        system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        output = self.prompt_expander(prompt, system_prompt=system_prompt, seed=seed)
        if output.status == False:
            print(f"Extending prompt failed: {output.message}")
            output_prompt = prompt
        else:
            output_prompt = output.prompt
        return output_prompt

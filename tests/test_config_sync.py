import dataclasses
import yaml
from cagf.model import ModelHParams

def test_default_config_model_section_matches_modelhparams_fields():
    cfg = yaml.safe_load(open('configs/default.yaml', encoding='utf-8'))
    valid_fields = {f.name for f in dataclasses.fields(ModelHParams)}
    yaml_keys = set(cfg['model'].keys())
    unknown = yaml_keys - valid_fields
    assert not unknown, f'configs/default.yaml has fields ModelHParams does not accept: {unknown}'
    ModelHParams(**cfg['model'])
if __name__ == '__main__':
    test_default_config_model_section_matches_modelhparams_fields()
    print('Config/ModelHParams sync test passed.')
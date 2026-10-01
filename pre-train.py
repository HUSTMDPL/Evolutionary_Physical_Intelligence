





from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode = True


def _shared():
    name = "_epi_shared"
    if name not in sys.modules:
        path = Path(__file__).resolve().with_name("latent space construction.py")
        if not path.is_file():
            raise FileNotFoundError(path)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load shared implementation: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def main(argv=None):
    return _shared().training_entry_v13("joint", argv)


if __name__ == "__main__":
    main()

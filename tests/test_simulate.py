import pandas as pd
import pytest

from simulate import main, parse_args


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(f"k: [4]\nfail_rate: [0, 0.1]\ntrials: 3\nout: {tmp_path / 'out'}\n")
    return str(path)


def test_k_from_config(config):
    args = parse_args(["--config", config])
    assert args.k == [4] and args.fail_rate == [0, 0.1] and args.trials == 3


def test_cli_overrides_config(config):
    assert parse_args(["--config", config, "--k", "8"]).k == [8]


def test_missing_k_is_an_error():
    with pytest.raises(SystemExit):
        parse_args([])


def test_end_to_end(config, tmp_path):
    main(["--config", config])
    df = pd.read_csv(tmp_path / "out" / "results.csv")
    assert len(df) == 2 * 3
    assert {"k", "fail_rate", "trial", "inter_mean", "intra_min",
            "edge_ports_min", "agg_ports_mean", "core_ports_min"} <= set(df.columns)
    assert (tmp_path / "out" / "config.json").exists()

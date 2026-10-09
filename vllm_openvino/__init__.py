import logging
from logging.config import dictConfig

from vllm.logger import DEFAULT_LOGGING_CONFIG


def register():
    """Register OpenVINO."""
    return "vllm_openvino.platform.OpenVinoPlatform"


class _ProcessNameFilter(logging.Filter):
    """vLLM installs the record factory that sets vllm_process_name only after
    plugins load; records emitted earlier would fail to format."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "vllm_process_name"):
            record.vllm_process_name = record.processName
        return True


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config = {**DEFAULT_LOGGING_CONFIG}

    # Copy the vLLM logging configurations
    config["formatters"]["vllm_openvino"] = DEFAULT_LOGGING_CONFIG["formatters"][
        "vllm"]

    handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
    handler_config["formatter"] = "vllm_openvino"
    handler_config["filters"] = ["vllm_openvino_process_name"]
    # vLLM later deep-copies DEFAULT_LOGGING_CONFIG, so the filter must live there.
    DEFAULT_LOGGING_CONFIG["filters"] = {
        "vllm_openvino_process_name": {"()": _ProcessNameFilter}}
    config["filters"] = DEFAULT_LOGGING_CONFIG["filters"]
    config["handlers"]["vllm_openvino"] = handler_config

    logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
    logger_config["handlers"] = ["vllm_openvino"]
    config["loggers"]["vllm_openvino"] = logger_config

    dictConfig(config)


_init_logging()
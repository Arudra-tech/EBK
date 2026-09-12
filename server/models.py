"""Pydantic schemas shared across the server."""

from pydantic import BaseModel, Field


class DeployConfig(BaseModel):
    runtime: str = "pytorch"
    precision: str = "fp32"
    resolution: int = 640
    batch_size: int = 1

    def label(self) -> str:
        return f"{self.precision.upper()}+{'TensorRT' if self.runtime == 'tensorrt' else 'PyTorch'} @{self.resolution} b{self.batch_size}"

    def key(self) -> tuple:
        return (self.runtime, self.precision, self.resolution, self.batch_size)


class Slo(BaseModel):
    target_latency_ms: float = Field(25.0, gt=0)
    max_accuracy_loss_pp: float = Field(0.5, ge=0)

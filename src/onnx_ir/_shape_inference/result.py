# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import dataclasses
from enum import Enum


class InferenceStatus(Enum):
    SUCCESS = 0
    FAILURE = 1
    UNSUPPORTED = 2


@dataclasses.dataclass
class InferenceResult:
    status: InferenceStatus
    reason: str | None = None

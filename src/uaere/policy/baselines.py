from uaere.types import ModelVariant, PolicyVector


def always_l3(device_id: str = "stm32l476") -> PolicyVector:
    return PolicyVector(tau1=0.0, tau2=0.0, tau3=0.0, device_id=device_id)


def always_l0(device_id: str = "stm32l476") -> PolicyVector:
    return PolicyVector(tau1=1.1, tau2=1.2, tau3=1.3, device_id=device_id)


def energy_then_classify(device_id: str = "stm32l476") -> PolicyVector:
    """Stand-in: low tau1 so L1 almost always, no L3. Used only as a named baseline."""
    return PolicyVector(
        tau1=0.05,
        tau2=0.05,
        tau3=1.1,
        device_id=device_id,
        model=ModelVariant.TINY,
    )

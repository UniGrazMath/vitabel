from vitabel.utils.helpers import match_object


def test_match_object():
    class Container:
        pass

    container = Container()
    container.a = 1
    container.b = 2
    container.c = 3
    container.metadata = {"name": "test", "version": "0.1"}

    assert match_object(container)
    assert match_object(container, a=1)
    assert match_object(container, a=1, b=2, c=3)
    assert not match_object(container, x=42)
    assert not match_object(container, a=1, c=3, x=42)
    assert not match_object(container, a=1, c=42)
    assert match_object(container, metadata={"name": "test"})
    assert match_object(container, metadata={"name": "test", "version": "0.1"})
    assert not match_object(container, metadata={"name": "test", "version": "0.2"})
    assert not match_object(container, metadata={"name": "test", "x": 42})

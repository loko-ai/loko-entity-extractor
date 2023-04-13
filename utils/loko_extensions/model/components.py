from typing import List

from loko_extensions.utils.pathutils import find_path
import json


class Input:
    """
        A component's input. Components can have multiple inputs.

        Example:

            >>> input1 = Input(id="input1", label="input1", to="output")
            >>> input2 = Input(id="input2", label="input2", to="output")
            >>> comp1 = Component(name="comp1", inputs=[input1, input2])
            >>> save_extensions([comp1])

        Args:
            id (str): The name of the input point.
            label (str): The visualized name. By default, it is set to the id value.
            service (str): Represents the path to the linked service. Default: `""`
            to (str): The name of the connected output.
        """
    def __init__(self, id, label=None, service=None, to="output"):

        self.id = id
        self.label = label or id
        self.service = service or ""
        self.to = to


class Output:
    """
        A component's output. Components can have multiple outputs.

        Example:

            >>> output1 = Output(id="output1", label="output1")
            >>> output2 = Output(id="output", label="output2")
            >>> comp1 = Component(name="comp1", outputs=[output1, output2])
            >>> save_extensions([comp1])

        Args:
            id (str): The name of the output point.
            label (str): The visualized name. By default, it is set to the id value.
        """
    def __init__(self, id, label=None):
        self.id = id
        self.label = label or id


class Arg:
    """
            A component's argument. Arguments are used to configure the block's parameters.
                See also: :py:meth:`~loko_extensions.model.components.Select`,
                :py:meth:`~loko_extensions.model.components.Dynamic`,
                :py:meth:`~loko_extensions.model.components.MultiKeyValue`,
                :py:meth:`~loko_extensions.model.components.AsyncSelect`.

            Example:

                >>> model_name = Arg(name="model_name", type="text", label="Model Name", helper="Helper text")
                >>> train = Arg(name="train", type="boolean", label="Train Model", description="Helper text")
                >>> comp1 = Component(name="comp1", args=[model_name, train])
                >>> save_extensions([comp1])

            Args:
                name (str): The name of the parameter.
                type (str): The parameter's type. Available types are: "text", "boolean", "number", "path", "files",
                    "directories", "code", "password", "email", "area". Default: `"text"`
                label (str): The visualized name. By default, it is set to the name value.
                helper (str): The explanation of the parameter usage.
                description (str): The explanation of the parameter usage. In this case it'll be displayed by clicking
                    on the "i" icon.
                group (str): The name of the parameter's section. It's used to divide parameters into groups.
                value: The default value of the parameter.
                required (bool): `True` if the parameter is required. Default: `False`
            """
    def __init__(self, name, type="text", label=None, helper="", description="", group="", value=None, required=False):
        self.name = name
        self.type = type
        self.label = label or name
        self.helper = helper
        self.group = group
        self.value = value
        self.description = description
        self.validation = dict(required="Required field") if required else None

    def to_dict(self):
        return self.__dict__


class Select(Arg):
    """
            A component's argument. Select is used to show a list of available options to configure the block's parameter.
                See also: :py:meth:`~loko_extensions.model.components.Arg`,
                :py:meth:`~loko_extensions.model.components.Dynamic`,
                :py:meth:`~loko_extensions.model.components.MultiKeyValue`,
                :py:meth:`~loko_extensions.model.components.AsyncSelect`.

            Example:

                >>> task = Select(name="task", label="Task", options=["sentiment analysis", "text generation",
                >>>               "question answering"])
                >>> comp1 = Component(name="comp1", args=[task])
                >>> save_extensions([comp1])

            Args:
                name (str): The name of the parameter.
                options (list): The list of available parameter's options.
                label (str): The visualized name. By default, it is set to the name value.
                helper (str): The explanation of the parameter usage.
                description (str): The explanation of the parameter usage. In this case it'll be displayed by clicking
                    on the "i" icon.
                group (str): The name of the parameter's section. It's used to divide parameters into groups.
                value: The default value of the parameter.
                required (bool): `True` if the parameter is required. Default: `False`
                """
    def __init__(self, name, options, label=None, helper="", description="", group="", value=None, required=False):
        super().__init__(name, "select", label, helper, description, group, value, required)
        self.options = options

class Dynamic(Arg):
    """
            A component's argument. Dynamic is used to dynamically show a parameter's configuration.
                See also: :py:meth:`~loko_extensions.model.components.Arg`,
                :py:meth:`~loko_extensions.model.components.Select`,
                :py:meth:`~loko_extensions.model.components.MultiKeyValue`,
                :py:meth:`~loko_extensions.model.components.AsyncSelect`.

            Example:
                >>> task = Select(name="task", label="Task", group="Task Settings", options=["sentiment analysis", "text generation", "question answering"])
                >>> max_length = Dynamic(name="max_length", label="Max Length", dynamicType="number", parent="task", group="Task Settings", value=30, condition="{parent}==='text generation'")
                >>> comp1 = Component(name='comp1', args=[task, max_length])
                >>> save_extensions([comp1])

            Args:
                name (str): The name of the parameter.
                parent (str): The name of the parameter it depends on.
                condition (str): The parameter will be displayed only when this condition is verified. The programming
                    language used in this case is JavaScript.
                label (str): The visualized name. By default, it is set to the name value.
                dynamicType (str): The parameter's type. Available types are: "text", "boolean", "number", "path",
                    "files", "directories", "code", "password", "email", "area", "select". Default: `"text"`
                options (list): The list of available parameter's options. Use this parameter only if dynamicType is
                    "select". Default: None
                helper (str): The explanation of the parameter usage.
                description (str): The explanation of the parameter usage. In this case it'll be displayed by clicking
                    on the "i" icon.
                group (str): The name of the parameter's section. It's used to divide parameters into groups.
                value: The default value of the parameter.
                required (bool): `True` if the parameter is required. Default: `False`
                """
    def __init__(self, name, parent, condition, label=None, dynamicType="text", options=None, url=None,
                 helper="", description="", group="", value=None, required=False):
        super().__init__(name, "dynamic", label, helper, description, group, value, required)
        self.parent = parent
        self.condition = condition
        self.dynamicType = dynamicType
        self.options = options
        self.url = url

class MKVField:
    """
            A MultiKeyValue field.
                See also: :py:meth:`~loko_extensions.model.components.MultiKeyValue`.

            Example:
                >>> mkvfields = [MKVField(name='field1', label='Field1', required=True),
                >>>              MKVField(name='field2', label='Field2')]
                >>> multikeyvalue = MultiKeyValue(name='multikeyvalue', label='MultiKeyValue', fields=mkvfields,
                >>>                               group='Advanced Args', required=True)
                >>> comp1 = Component(name='comp1', args=[multikeyvalue])
                >>> save_extensions([comp1])

                and returns a list of dictionaries:

                >>> [{'field1': 'inputvalue1', 'field2': 'inputvalue2', 'id': '7a6d409c-38e2-482f-9e28-be48b2ccf1ef'}]

            Args:
                name (str): The name of the parameter.
                label (str): The visualized name. By default, it is set to the name value.
                required (bool): Set it to `True` if the parameter is required.
                """
    def __init__(self, name, label=None, required=False):
        self.name = name
        self.placeholder = label or name
        self.validation = dict(required="Required field") if required else None



class MultiKeyValue(Arg):
    """
            A component's argument. MultiKeyValue is used to set lists of variable length.
                See also: :py:meth:`~loko_extensions.model.components.Arg`,
                :py:meth:`~loko_extensions.model.components.Select`,
                :py:meth:`~loko_extensions.model.components.Dynamic`,
                :py:meth:`~loko_extensions.model.components.AsyncSelect`.

            Example:
                >>> mkvfields = [MKVField(name='field1', label='Field1', required=True),
                >>>              MKVField(name='field2', label='Field2')]
                >>> multikeyvalue = MultiKeyValue(name='multikeyvalue', label='MultiKeyValue', fields=mkvfields,
                >>>                               group='Advanced Args', required=True)
                >>> comp1 = Component(name='comp1', args=[multikeyvalue])
                >>> save_extensions([comp1])

                and returns a list of dictionaries:

                >>> [{'field1': 'inputvalue1', 'field2': 'inputvalue2', 'id': '7a6d409c-38e2-482f-9e28-be48b2ccf1ef'}]

            Args:
                name (str): The name of the parameter.
                fields (List[MKVField]): The list of fields.
                label (str): The visualized name. By default, it is set to the name value.
                helper (str): The explanation of the parameter usage.
                description (str): The explanation of the parameter usage. In this case it'll be displayed by clicking
                    on the "i" icon.
                group (str): The name of the parameter's section. It's used to divide parameters into groups.
                value: The default value of the parameter.
                required (bool): `True` if the parameter is required. Default: `False`
                """

    def __init__(self, name, fields, label=None, helper="", description="", group="", value=None, required=False):
        super().__init__(name, "multiKeyValue", label, helper, description, group, value, required)
        self.fields = fields

    def to_dict(self):
        d = super().to_dict()
        d['fields'] = [f.__dict__ for f in d['fields']]
        return d


class AsyncSelect(Arg):
    """
            A component's argument. AsyncSelect is used to show a list of available options to configure the block's
                parameter. Unlike Select Arg it takes options from the result of a GET request.
                See also: :py:meth:`~loko_extensions.model.components.Arg`,
                :py:meth:`~loko_extensions.model.components.Select`,
                :py:meth:`~loko_extensions.model.components.Dynamic`
                :py:meth:`~loko_extensions.model.components.MultiKeyValue`.

            Example:
                >>> task = AsyncSelect(name='task', label='Task', url='http://localhost:9999/routes/first_project/tasks')
                >>> comp1 = Component(name="comp1", args=[task])
                >>> save_extensions([comp1])

            Args:
                name (str): The name of the parameter.
                url (str): GET request's url.
                label (str): The visualized name. By default, it is set to the name value.
                helper (str): The explanation of the parameter usage.
                description (str): The explanation of the parameter usage. In this case it'll be displayed by clicking
                    on the "i" icon.
                group (str): The name of the parameter's section. It's used to divide parameters into groups.
                value: The default value of the parameter.
                required (bool): `True` if the parameter is required. Default: `False`
                """

    def __init__(self, name, url, label=None, helper="", description="", group="", value=None, required=False):
        super().__init__(name, "asyncSelect", label, helper, description, group, value, required)
        self.url = url

class Component:
    """
        A customized Loko component.

        Example:
            >>> comp1 = Component(name="comp1")
            >>> save_extensions([comp1])

        Args:
            name (str): The name of the component.
            description (str): The explanation of the component.
            group (str): The section name in the sidebar. Default: `Custom`
            inputs (List[Input]): The list of the component's inputs. Default: `[Input("input")]`
            outputs (List[Output]): The list of the component's outputs. Default: `[Output("output")]`
            args (List[Arg]): The list of the component's arguments. Default: `None`
            trigger (bool): Set to `True` to enable start. Default: `False`
            configured (bool): `False` if configurations are required. Default: `True`
            icon (str): The component's icon. Available values: `react-icons/ri <https://react-icons.github.io/react-icons/icons?name=ri>`_.
                        Default: `"RiCheckboxBlankCircleFill"`

        """
    def __init__(self, name, description="", group="Custom", inputs=None, outputs=None, args=None, trigger=False,
                 configured=True, icon="RiCheckboxBlankCircleFill", events=None):
        self.name = name
        self.description = description
        self.group = group
        self.inputs = inputs or [Input("input")]
        self.outputs = outputs or [Output("output")]
        self.args = args or []
        self.icon = icon
        self.click = "Send message" if trigger else None
        self.events = events

        self.configured = configured

    def to_dict(self):
        values = {x.name: x.value for x in self.args if x.value}
        options = dict(values=values, args=[x.to_dict() for x in self.args])
        return dict(name=self.name, description=self.description, group=self.group,
                    icon=self.icon, click=self.click,
                    events=self.events,
                    configured=self.configured,
                    inputs=[x.__dict__ for x in self.inputs],
                    outputs=[x.__dict__ for x in self.outputs], options=options)


def save_extensions(comps, path="extensions"):
    """
        Save a list of components into the components.json file.
    """
    output_path = find_path(path)
    output = output_path / "components.json"
    with output.open("w") as o:
        json.dump([comp.to_dict() for comp in comps], o, indent=1)
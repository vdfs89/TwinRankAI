"""Schemas Pandera para os 3 arquivos do RetailRocket.

Funciona como o "data validation gate" antes do treino: schema, tipos,
faixas de valores e nulidade são checados aqui, falhando cedo se os dados
brutos mudarem de formato.
"""

import pandera as pa
from pandera.typing import Series


class EventsSchema(pa.DataFrameModel):
    """Schema de events.csv — interações visitor-item."""

    timestamp: Series[int] = pa.Field(ge=0)
    visitorid: Series[int] = pa.Field(ge=0)
    event: Series[str] = pa.Field(isin=["view", "addtocart", "transaction"])
    itemid: Series[int] = pa.Field(ge=0)
    transactionid: Series[float] = pa.Field(nullable=True)

    class Config:  # noqa: D106
        coerce = True
        strict = False


class ItemPropertiesSchema(pa.DataFrameModel):
    """Schema de item_properties.csv — propriedades hasheadas dos itens."""

    timestamp: Series[int] = pa.Field(ge=0)
    itemid: Series[int] = pa.Field(ge=0)
    property: Series[str]
    value: Series[str]

    class Config:  # noqa: D106
        coerce = True
        strict = False


class CategoryTreeSchema(pa.DataFrameModel):
    """Schema de category_tree.csv — hierarquia de categorias."""

    categoryid: Series[int] = pa.Field(ge=0)
    parentid: Series[float] = pa.Field(nullable=True)

    class Config:  # noqa: D106
        coerce = True
        strict = False


def validate_events(df: "pa.typing.DataFrame") -> "pa.typing.DataFrame":
    """Valida events.csv contra o schema; levanta SchemaError se inválido."""
    return EventsSchema.validate(df, lazy=True)


def validate_item_properties(df: "pa.typing.DataFrame") -> "pa.typing.DataFrame":
    """Valida item_properties.csv contra o schema; levanta SchemaError se inválido."""
    return ItemPropertiesSchema.validate(df, lazy=True)


def validate_category_tree(df: "pa.typing.DataFrame") -> "pa.typing.DataFrame":
    """Valida category_tree.csv contra o schema; levanta SchemaError se inválido."""
    return CategoryTreeSchema.validate(df, lazy=True)

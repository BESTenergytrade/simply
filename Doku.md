# Doku.md

```
In the Doku.md the functionalities of simply should be summarized. Each script is mentioned shortly.
```

## Introduction

```
Simply introduces the user to a generic energy market. 
The prosumers are actors who generate and calculate orders of different types. 
These types can be +1 (Bids) and -1 (Asks). These orders are being sent to the market. 
The market keeps an overview over the different orders per timestamp of each actor and
can accept the Orders and clears the market. 
The PowerNetwork describes the connections of a power grid where nodes can be
active participants or distributing power stations. The edges are abstracted connections between those.
Edges can have weights which represent power fees.
The scenario in itself is random in terms of who is participating with who. For the simulation main.py 
specifies the start and intervall of the simulation
```
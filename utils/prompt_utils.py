import numpy as np
import torch

from utils.vector_utils import (
    MS_TO_MPH,
    VELOCITY_MS_SCALE,
    EgoField,
    PedestrianField,
    VectorObservation,
    VehicleField,
    angles_deg_and_distances,
    control_to_pedals,
    distance_to_junction,
    get_tl_state,
    object_direction,
    pedestrian_filter_flags,
    route_angles_from_route_desc,
    side,
    traveling_angle_deg_from_pedestrian_desc,
    traveling_angle_deg_from_vehicle_desc,
    vehicle_filter_flags,
    xy_from_pedestrian_desc,
    xy_from_route_desc,
    xy_from_vehicle_desc,
)


def make_observation_prompt(  
    obs,
    att=None,
    vehicle_descriptions=None,
    pedestrian_descriptions=None,
    agent_id=False,
    attention=False,
):
    if isinstance(obs, dict):
        obs = VectorObservation(**obs)
    if att is not None:
        assert obs.vehicle_descriptors.shape[0] == att["vehicles"].shape[0]
        assert obs.pedestrian_descriptors.shape[0] == att["pedestrians"].shape[0]
    if vehicle_descriptions is not None:
        assert torch.sum(obs.vehicle_descriptors[:, VehicleField.ACTIVE]) == len(
            vehicle_descriptions
        )
    if pedestrian_descriptions is not None:
        active_peds = int(
            torch.sum(obs.pedestrian_descriptors[:, PedestrianField.ACTIVE])
        )
        if len(pedestrian_descriptions) > active_peds:
            pedestrian_descriptions = pedestrian_descriptions[:active_peds]
        assert active_peds == len(pedestrian_descriptions)

    vehicle_flags = vehicle_filter_flags(obs.vehicle_descriptors)

    # dynamic vehicles
    vehicle_flags &= obs.vehicle_descriptors[:, VehicleField.DYNAMIC] == 1
    vehicles = obs.vehicle_descriptors[vehicle_flags]
    vehicle_attn = (
        att["vehicles"][vehicle_flags] if att is not None and attention else None
    )
    visible_vehicle_descriptions = (
        [vehicle_descriptions[i] for i in torch.where(vehicle_flags)[0]]
        if vehicle_descriptions is not None and agent_id
        else None
    )

    ped_flags = pedestrian_filter_flags(obs.pedestrian_descriptors)
    pedestrians = obs.pedestrian_descriptors[ped_flags]

    pedestrian_attn = (
        att["pedestrians"][ped_flags] if att is not None and attention else None
    )
    visible_ped_descriptions = (
        [pedestrian_descriptions[i] for i in torch.where(ped_flags)[0]]
        if pedestrian_descriptions is not None and agent_id
        else None
    )

    distances_vehicles, angular_vehicles = angles_deg_and_distances(
        xy_from_vehicle_desc(vehicles)
    )
    vehicle_traveling_direction = traveling_angle_deg_from_vehicle_desc(vehicles)
    pedestrian_traveling_direction = traveling_angle_deg_from_pedestrian_desc(
        pedestrians
    )

    distances_peds, angular_peds = angles_deg_and_distances(
        xy_from_pedestrian_desc(pedestrians)
    )
    crossing = pedestrians[:, PedestrianField.CROSSING] > 0.0

    route_xy = xy_from_route_desc(obs.route_descriptors)

    distances_route, angular_route = angles_deg_and_distances(route_xy)
    tl_state, tl_distance = get_tl_state(obs.route_descriptors)

    vehicle_lines = []
    if vehicle_attn is None:
        vehicle_ordering = range(len(vehicles))
    else:
        vehicle_ordering = np.argsort(vehicle_attn)[::-1]
    for i in vehicle_ordering:
        if agent_id:
            assert visible_vehicle_descriptions is not None
            v_id = visible_vehicle_descriptions[i]["id"]
            model = visible_vehicle_descriptions[i]["model"].split("-")[0].lower()
            color = visible_vehicle_descriptions[i]["color"][len("vehicle_") :]
        vehicle_lines.append(
            f"A moving car"  # agent
            + (
                f" ({color} {model}, id={v_id})" if agent_id else ""
            )  # color and model and id
            + f"; Angle in degrees: {angular_vehicles[i]:.2f}; Distance: {distances_vehicles[i]:.2f}m; Direction of travel: {object_direction(vehicle_traveling_direction[i])};"  # angle, distance, direction
            + (f" My attention: {int(vehicle_attn[i]*100):d}%" if attention else "")
        )

    pedestrian_lines = []
    if pedestrian_attn is None:
        pedestrian_ordering = range(len(pedestrians))
    else:
        pedestrian_ordering = np.argsort(pedestrian_attn)[::-1]
    for i in pedestrian_ordering:
        if agent_id:
            assert visible_ped_descriptions is not None
            p_id = visible_ped_descriptions[i]["id"]
        pedestrian_lines.append(
            f"A pedestrian"
            + (f" (id={p_id})" if agent_id else "")
            + f"; Angle in degrees: {angular_peds[i]:.2f}; Distance: {distances_peds[i]:.2f}m; Direction of travel: {object_direction(pedestrian_traveling_direction[i])}; Crossing: {crossing[i]};"  # angle, distance, direction, crossing
            + (f" My attention: {int(pedestrian_attn[i]*100):d}%" if attention else "")
        )

    input_prompt = (
        f"\nI'm observing {len(vehicles)} cars and {len(pedestrians)} pedestrians.\n"
    )
    input_prompt += "\n".join(vehicle_lines + pedestrian_lines) + "\n"

    route_angles = route_angles_from_route_desc(obs.route_descriptors)

    angle_diffs = torch.diff(route_angles)
    angle_diffs[angle_diffs > 180] -= 360
    angle_diffs[angle_diffs < -180] += 360
    total_turn_right = torch.sum(angle_diffs[angle_diffs > 0])
    total_turn_left = torch.sum(angle_diffs[angle_diffs < 0])
    is_roundabout = abs(total_turn_left) > 30 and abs(total_turn_right) > 30

    current_speed_mph = (
        obs.ego_vehicle_descriptor[EgoField.SPEED]
    )

    input_prompt += f"My current speed is {current_speed_mph:.2f} m/s"

    return input_prompt


def make_action_prompt(output, precise_steering=True):
    control_longitudinal, control_lateral = output.action_distribution.mean()
    accelerator_pedal_pct, brake_pressure_pct = control_to_pedals(control_longitudinal)

    vy = accelerator_pedal_pct - brake_pressure_pct  

    vx = control_lateral  

    input_prompt = f"""
- vx {vx:.2f}m/s
- vy {vy:.2f}m/s
- vz {0:.2f}m/s
"""
    if precise_steering:
        input_prompt += f"""- Going to vx {abs(vx):.2f}% and vy {abs(vy):.2f}."""
    else:
        input_prompt += """- Going straight on."""

    return input_prompt

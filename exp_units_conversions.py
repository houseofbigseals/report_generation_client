

"""

This module contains experimental data coefficients for
calculating what value of currents (Ired, Iwhite) gives us
required values of (FAR, FR/FW) and vice versa

All here is very unstable and depends on configuration of
experimental device, be careful

"""

volume = 80.0  # fitotrone volume in litres
raw_to_dry = 0.08  # conversion factor from raw plants weight to dry weight
ppmv_to_mgco2 = 1.8  # conversion factor from ppmv CO2 to mgCO2/m3
surface = 0.19  # in m2 - surface of lighted crops
control_surface = 0.31  # m2 - surface of control stand
surface_to_volume = 0.45  # in m3/m2
mg_co2_to_kg_dry_mass = 0.68*0.001*0.001 # in kg of dry mass / mg CO2 assimilated
mg_co2_to_kg_raw_mass = 8.5*0.001*0.001 # in kg of dry mass / mg CO2 assimilated
# when water coefficient is 0.08
ppfd_to_kw = 0.2*0.001  # kW / (mkmol/m2*sec)
price_of_volume = 45.2  # kg_of_equiv_mass / m3
price_of_power = 114.0  # kg_of_equiv_mass / kW
old_price_of_volume = 0.28
old_price_of_power = 0.72
a1 = 2.0877467
b1 = 3.6243109
a2 = 2.64379709
b2 = -0.53008089

def red_far_by_curr(Ir):
    # this constants are determined from experiment
    a1 = 2.0877467
    b1 = 3.6243109
    return a1*Ir + b1


def white_far_by_curr(Iw):
    # this constants are determined from experiment
    a2 = 2.64379709
    b2 = -0.53008089
    return a2*Iw + b2


def currents_from_newcoords(A, B):
    """
    :param A: A is FAR = FAR_red + FAR_white in mkmoles
    :param B: B is FAR_red / FAR_white
    return: (I_red, I_white)
    """
    # # this constants are determined from experiment
    # # for far lamp position
    # a1 = 1.77451454
    # b1 = 5.52067992
    # a2 = 2.40069694
    # b2 = 0.24050309

    # this constants are determined from experiment
    # for near lamp position
    # [2.0877467  3.6243109]
    # [2.64379709 - 0.53008089]
    a1 = 2.0877467
    b1 = 3.6243109
    a2 = 2.64379709
    b2 = -0.53008089

    if B != 0:

        # formula was gotten analytically
        Ir = ((A*B)/(B+1) - b1)/a1
        Iw = (A/(B+1) - b2)/a2

        # check if currents less then 10:
        # TODO check if it really good decision
        if Ir < 10:
            Ir = 10
        if Iw < 10:
            Iw = 10

        return Ir, Iw

    else:
        Ir = 10 # because our lamp cant do less than 10 mA
        z = 18.5809333333
        Iw = (A - z - b2)/a2

        return Ir, Iw


# functions to calculate different Q for moon from raw -dCO2/dt

def Q(dC, E, weight):
    global volume
    global raw_to_dry
    global ppmv_to_mgco2
    # convert from ppmv/sec to mg CO2/(m3*sec)
    dCC = ppmv_to_mgco2 * dC
    # then convert from 1m3=1000litres to our volume
    dCC = (volume/1000) * dCC
    # convert weight from raw to dry
    dry_weight = weight*raw_to_dry
    # then calculate Q and divide it to mean weight
    # return ((0.28/1.9) * dCC + (0.72/0.0038) * (dCC / E)) / dry_weight
    return ((0.28 / 1.9) * dCC + (0.72 / 0.0038) * (dCC / E))


def rQ(dC, E, weight):
    global volume
    global raw_to_dry
    global ppmv_to_mgco2
    # convert from ppmv/sec to mg CO2/(m3*sec)
    dCC = ppmv_to_mgco2 * dC
    # then convert from 1m3=1000litres to our volume
    dCC = (volume/1000) * dCC
    # convert weight from raw to dry
    # dry_weight = weight*raw_to_dry
    # then calculate Q and divide it to mean weight
    # return ((0.28/1.9) * dCC + (0.72/0.0038) * (dCC / E)) / dry_weight
    return (0.28 / dCC) + (0.72 * (E / dCC))


def FE(dC, E, weight):
    global volume
    global raw_to_dry
    global ppmv_to_mgco2
    # convert from ppmv/sec to mg CO2/(m3*sec)
    dCC = ppmv_to_mgco2 * dC
    # then convert from 1m3 to our volume
    dCC = (volume/1000) * dCC
    # convert weight from raw to dry
    dry_weight = weight*raw_to_dry
    # then calculate Q and divide it to mean weight
    # return (dCC / E) / (dry_weight * 0.0038)
    return dCC / E


def F(dC, weight):
    global volume
    global raw_to_dry
    global ppmv_to_mgco2
    # convert from ppmv/sec to mg CO2/(m3*sec)
    dCC = ppmv_to_mgco2 * dC
    # then convert from 1m3 to our volume
    dCC = (volume/1000) * dCC
    # convert weight from raw to dry
    dry_weight = weight*raw_to_dry
    # then calculate Q and divide it to mean weight
    # return dCC / dry_weight
    return dCC


def raw_intQ(dC, E, dT):
    # dC - first derivative of co2 concentration in ppnmv/sec
    # E - light intencity im mkmoles/m2*sec
    # dT - time period of measure im sec
    global volume
    global surface
    global ppmv_to_mgco2
    # convert from ppmv/sec to mg CO2/(m3*sec)
    dCC = ppmv_to_mgco2 * dC
    # then convert from 1m3 to our volume
    dCC = (volume/1000.0) * dCC
    # now dCC is mgCO2/sec in our volume
    V = (surface_to_volume * surface)  # effective volume of crop in m3
    Prod = mg_co2_to_kg_raw_mass*dCC*dT  # productivity of crops in kg
    I = E* ppfd_to_kw   # light power converted to kW
    Qi = price_of_volume * V / Prod + price_of_power * I * surface / Prod
    return Qi


def dry_intQ(dC, E, dT):
    # dC - first derivative of co2 concentration in ppnmv/sec
    # E - light intencity im mkmoles/m2*sec
    # dT - time period of measure im sec
    global volume
    global surface
    global ppmv_to_mgco2
    global surface_to_volume
    # convert from ppmv/sec to mg CO2/(m3*sec)
    dCC = ppmv_to_mgco2 * dC
    # then convert from 1m3 to our volume
    dCC = (volume/1000.0) * dCC
    # now dCC is mgCO2/sec in our volume
    V = (surface_to_volume * surface)  # effective volume of crop in m3
    Prod = mg_co2_to_kg_dry_mass*dCC*dT  # productivity of crops in kg
    # dT must be in sec
    I = E * ppfd_to_kw  # light power converted to kW
    Qi = price_of_volume * V / Prod + price_of_power * I * surface / Prod
    return Qi


def final_intQ(E, Prod, mode):
    # Prod  - is dM in grams
    # E - light intencity im mkmoles/m2*sec
    global volume
    global surface
    global control_surface
    global ppmv_to_mgco2
    global surface_to_volume
    global ppfd_to_kw
    Prod = Prod*0.001  # translate to kg
    # now dCC is mgCO2/sec in our volume
    if mode == "exp":
        V = (surface_to_volume * surface)  # effective volume of crop in m3
    if mode == "control":
        V = (surface_to_volume * control_surface)  # effective volume of crop in m3

    I = E * ppfd_to_kw  # light power converted to kW / m2
    Qf = price_of_volume * V / Prod + price_of_power * I * surface / Prod
    return Qf


if __name__ == "__main__":
    # print(currents_from_newcoords(0, 0))
    # print(currents_from_newcoords(500, 0.125))
    # print(currents_from_newcoords(500, 0.25))
    # print(currents_from_newcoords(500, 0.5))
    # print(currents_from_newcoords(500, 0.75))
    # print(currents_from_newcoords(500, 1))
    print(currents_from_newcoords(500, 1.5))
    # print(currents_from_newcoords(500, 1.75))
    # print(currents_from_newcoords(500, 2))
    # print(currents_from_newcoords(500, 2.25))
    # print(currents_from_newcoords(500, 6))
    # print(currents_from_newcoords(500, 40))
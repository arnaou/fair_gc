import numpy as np


def retrieve_consts(property_tag):
    """

    :param property_tag:
    :return:
    """
    n_const = None
    list_of_const = []

    if property_tag == 'Omega':
        n_const = 3
        list_of_const = [0.9080 ,0.1055, 1.002]
    elif property_tag == 'Tc':
        n_const = 1
        list_of_const = [400]
    elif property_tag == 'Vc':
        n_const = 1
        list_of_const = [20]
    elif property_tag == 'Pc':
        n_const = 1
        list_of_const = [0.1347]

    return n_const, list_of_const



def linearize_gc(y,const,property_tag):
    y_lin = None
    if property_tag == 'Tc':
        y_lin = np.exp(y/const)
    elif property_tag == 'Pc':
        y_lin = ((y-0.0519)**(-0.5))-const
    elif property_tag in ['Vc', 'HCOM']:
        y_lin = y - const
    elif property_tag == 'Omega':
        y_lin = np.exp(y/const[0])**(const[1])-const[2]

    return y_lin


def predict_gc(params, X, property):
    y_pred=None
    if property in ['Tc', 'Tb', 'Tm']:
        rhs = np.matmul(X, params[1:])
        y_pred = params[0] * np.log(rhs)
    elif property == 'Pc':
        rhs = np.matmul(X, params[1:])
        y_pred = 1/((rhs+params[0])**2)+0.0519
    elif property in ['Vc', 'HCOM'] :
        rhs = np.matmul(X, params[1:])
        y_pred = rhs+params[0]
    elif property == 'Omega':
        rhs = np.matmul(X, params[3:])
        y_pred = params[0] * np.log((rhs + params[2]) ** (1 / params[1]))

    return y_pred


def Fobj_fog(params1, X1, y_target, property, output):

    params = params1.tolist()
    X = X1

    y_pred = predict_gc(params, X, property)

    residuals = y_target - y_pred
    if output == 'res':
        out = residuals.ravel()
    elif output == 'sse':
        out = np.sum(residuals ** 2)

    return out

def Fobj_sog(params2, X2, y_target, property, X1, params1, output):

    params =np.insert(params2, 0, params1)
    X = np.concatenate((X1, X2), axis=1)

    y_pred = predict_gc(params, X, property)

    residuals = y_target - y_pred

    if output == 'res':
        out = residuals.ravel()
    elif output == 'sse':
        out = np.sum(residuals ** 2)

    return out

def Fobj_tog(params3, X3, y_target, property, X1, params1, X2, params2, output):

    params = np.insert(params2, 0, params1)
    params = np.insert(params3, 0, params)

    X = np.concatenate((X1, X2, X3), axis=1)

    y_pred = predict_gc(params, X, property)

    residuals = y_target - y_pred

    if output == 'res':
        out = residuals.ravel()
    elif output == 'sse':
        out = np.sum(residuals ** 2)

    return out